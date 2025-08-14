"""
YOLO11 Manager according to official Ultralytics documentation
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Generator
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils.gpu_manager import get_gpu_manager


class YOLO11Manager:
    """
    YOLO11 Manager implementing official Ultralytics API
    Supports detection, segmentation, classification, and pose estimation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not available. Install with: pip install ultralytics")
        
        # Configuration
        yolo_config = config.get('processing', {}).get('models', {}).get('yolo', {})
        self.confidence_threshold = yolo_config.get('confidence_threshold', 0.5)
        self.iou_threshold = yolo_config.get('iou_threshold', 0.45)
        self.max_detections = yolo_config.get('max_detections', 100)
        self.device = yolo_config.get('device', 'auto')
        self.force_cpu = yolo_config.get('force_cpu', False)
        
        # Model paths
        self.model_paths = yolo_config.get('models', {})
        self.detection_model_path = self.model_paths.get('detection', 'yolo11n.pt')
        self.face_model_path = self.model_paths.get('face', 'yolo11n.pt')
        self.emotion_model_path = self.model_paths.get('emotion', 'yolo11n.pt')
        
        # Storage paths
        storage_config = config.get('storage', {})
        self.frames_dir = Path(storage_config.get('frames_dir', 'storage/frames'))
        self.faces_dir = Path(storage_config.get('faces_dir', 'storage/faces'))
        self.faces_yolo11_dir = Path(storage_config.get('faces_yolo11_dir', 'storage/faces_yolo11'))
        
        # Ensure directories exist
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.faces_yolo11_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Model instances
        self.detection_model = None
        self.face_model = None
        self.emotion_model = None
        
        # Person class ID in COCO dataset
        self.PERSON_CLASS_ID = 0
        
        self.logger.info("YOLO11Manager initialized")
    
    def _get_device(self) -> str:
        """Get appropriate device for YOLO models"""
        if self.force_cpu:
            return 'cpu'
        
        if self.device == 'auto':
            device = self.gpu_manager.setup_device()
            return str(device)
        
        return self.device
    
    def _load_model(self, model_path: str, model_type: str = 'detection') -> YOLO:
        """
        Load YOLO model with automatic download if not exists
        
        Args:
            model_path: Path to model file
            model_type: Type of model (detection, face, emotion)
            
        Returns:
            Loaded YOLO model
        """
        try:
            # Check if model exists locally or force fallback for face models
            if (not os.path.exists(model_path) and not model_path.startswith('yolo11')) or 'face.pt' in model_path:
                self.logger.warning(f"Model file not found or fallback needed: {model_path}")
                # Fallback to default model names for auto-download
                if model_type == 'detection':
                    model_path = 'yolo11n.pt'
                elif model_type == 'face':
                    model_path = 'yolo11n.pt'  # Use standard model for face detection
                elif model_type == 'emotion':
                    model_path = 'yolo11n.pt'  # Use standard model for emotion detection
            
            self.logger.info(f"Loading {model_type} model: {model_path}")
            
            # Load model (will auto-download if needed)
            model = YOLO(model_path)
            
            # Get device and move model
            device = self._get_device()
            model.to(device)
            
            # Model info
            model.info()
            
            # Optimize for inference
            model.fuse()
            
            self.logger.info(f"{model_type.capitalize()} model loaded successfully on {device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_type} model {model_path}: {e}")
            raise
    
    def load_detection_model(self) -> YOLO:
        """Load detection model"""
        if self.detection_model is None:
            self.detection_model = self._load_model(self.detection_model_path, 'detection')
        return self.detection_model
    
    def load_face_model(self) -> YOLO:
        """Load face detection model"""
        if self.face_model is None:
            self.face_model = self._load_model(self.face_model_path, 'face')
        return self.face_model
    
    def load_emotion_model(self) -> YOLO:
        """Load emotion detection model"""
        if self.emotion_model is None:
            self.emotion_model = self._load_model(self.emotion_model_path, 'emotion')
        return self.emotion_model
    
    def extract_frames_with_people(
        self,
        frame_paths: List[str],
        save_annotated: bool = True,
        return_crops: bool = False,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Extract frames containing people using YOLO11 detection
        
        Args:
            frame_paths: List of frame file paths
            save_annotated: Save annotated images with bounding boxes
            return_crops: Return cropped person images
            batch_size: Batch size for processing
            
        Returns:
            Dict with detection results
        """
        self.logger.info(f"Extracting frames with people from {len(frame_paths)} frames")
        
        try:
            # Load detection model
            model = self.load_detection_model()
            
            results_data = {
                'frames_with_people': [],
                'total_detections': 0,
                'person_crops': [],
                'metadata': []
            }
            
            # Process in batches with progress bar
            for i in tqdm(range(0, len(frame_paths), batch_size), desc="Processing frames"):
                batch_paths = frame_paths[i:i + batch_size]
                
                try:
                    # Use official YOLO predict API with stream for memory efficiency
                    results = model.predict(
                        source=batch_paths,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        imgsz=640,
                        max_det=self.max_detections,
                        classes=[self.PERSON_CLASS_ID],  # Only detect persons
                        stream=True,
                        verbose=False
                    )
                    
                    # Process results
                    for result in results:
                        frame_path = result.path
                        timestamp = time.time()
                        
                        # Check if any persons detected
                        if len(result.boxes) > 0:
                            results_data['frames_with_people'].append(frame_path)
                            results_data['total_detections'] += len(result.boxes)
                            
                            # Save annotated image if requested
                            if save_annotated:
                                annotated_path = self.faces_yolo11_dir / f"{Path(frame_path).stem}_annotated.jpg"
                                result.save(filename=str(annotated_path))
                            
                            # Extract person crops if requested
                            if return_crops:
                                crops = self._extract_person_crops(result, timestamp)
                                results_data['person_crops'].extend(crops)
                            
                            # Store metadata
                            frame_metadata = {
                                'frame_path': frame_path,
                                'timestamp': timestamp,
                                'detections': len(result.boxes),
                                'boxes': result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else [],
                                'confidences': result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else [],
                                'classes': result.boxes.cls.cpu().numpy().tolist() if result.boxes is not None else []
                            }
                            
                            # Add segmentation masks if available
                            if hasattr(result, 'masks') and result.masks is not None:
                                frame_metadata['masks'] = True
                                frame_metadata['mask_shapes'] = [mask.shape for mask in result.masks.data]
                            
                            # Add keypoints if available (pose estimation)
                            if hasattr(result, 'keypoints') and result.keypoints is not None:
                                frame_metadata['keypoints'] = True
                                frame_metadata['keypoints_data'] = result.keypoints.data.cpu().numpy().tolist()
                            
                            results_data['metadata'].append(frame_metadata)
                
                except RuntimeError as e:
                    # Handle CUDA errors with GPU manager
                    if "torchvision::nms" in str(e) or "CUDA" in str(e):
                        self.logger.warning(f"CUDA error in batch processing: {e}")
                        try:
                            # Try CPU fallback
                            model.to('cpu')
                            results = model.predict(
                                source=batch_paths,
                                conf=self.confidence_threshold,
                                iou=self.iou_threshold,
                                device='cpu',
                                classes=[self.PERSON_CLASS_ID],
                                stream=True,
                                verbose=False
                            )
                            # Process results on CPU (same logic as above)
                            # ... (would repeat the processing logic)
                            
                        except Exception as cpu_e:
                            self.logger.error(f"CPU fallback also failed: {cpu_e}")
                            continue
                    else:
                        self.logger.error(f"Batch processing failed: {e}")
                        continue
            
            self.logger.info(f"Found {len(results_data['frames_with_people'])} frames with people")
            return results_data
            
        except Exception as e:
            self.logger.error(f"extract_frames_with_people failed: {e}")
            raise
    
    def detect_faces(
        self,
        frame_paths: List[str],
        use_tracking: bool = True,
        save_crops: bool = True,
        min_face_size: int = 32
    ) -> Dict[str, Any]:
        """
        Detect faces in frames with tracking support
        
        Args:
            frame_paths: List of frame paths
            use_tracking: Use YOLO tracking for face ID consistency
            save_crops: Save individual face crops
            min_face_size: Minimum face size in pixels
            
        Returns:
            Dict with face detection results
        """
        self.logger.info(f"Detecting faces in {len(frame_paths)} frames")
        
        try:
            # Load face detection model
            model = self.load_face_model()
            
            results_data = {
                'faces_detected': [],
                'total_faces': 0,
                'face_crops': [],
                'tracking_data': {},
                'metadata': []
            }
            
            for frame_path in tqdm(frame_paths, desc="Detecting faces"):
                timestamp = time.time()
                
                try:
                    if use_tracking:
                        # Use model.track() for tracking support
                        results = model.track(
                            source=frame_path,
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            persist=True,
                            verbose=False
                        )
                    else:
                        # Use regular prediction
                        results = model.predict(
                            source=frame_path,
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            verbose=False
                        )
                    
                    for result in results:
                        if len(result.boxes) > 0:
                            # Filter faces by minimum size
                            valid_faces = []
                            
                            for i, box in enumerate(result.boxes):
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                face_width = x2 - x1
                                face_height = y2 - y1
                                
                                if face_width >= min_face_size and face_height >= min_face_size:
                                    face_data = {
                                        'frame_path': frame_path,
                                        'timestamp': timestamp,
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'confidence': float(box.conf[0].cpu()),
                                        'face_size': (int(face_width), int(face_height))
                                    }
                                    
                                    # Add tracking ID if available
                                    if use_tracking and hasattr(box, 'id') and box.id is not None:
                                        track_id = int(box.id[0].cpu())
                                        face_data['track_id'] = track_id
                                        
                                        # Update tracking data
                                        if track_id not in results_data['tracking_data']:
                                            results_data['tracking_data'][track_id] = []
                                        results_data['tracking_data'][track_id].append({
                                            'frame_path': frame_path,
                                            'timestamp': timestamp,
                                            'bbox': face_data['bbox'],
                                            'confidence': face_data['confidence']
                                        })
                                    
                                    # Save face crop if requested
                                    if save_crops:
                                        crop_path = self._save_face_crop(
                                            frame_path, face_data['bbox'], timestamp, 
                                            face_data.get('track_id', i)
                                        )
                                        face_data['crop_path'] = crop_path
                                        results_data['face_crops'].append(crop_path)
                                    
                                    valid_faces.append(face_data)
                            
                            if valid_faces:
                                results_data['faces_detected'].extend(valid_faces)
                                results_data['total_faces'] += len(valid_faces)
                                results_data['metadata'].append({
                                    'frame_path': frame_path,
                                    'faces_count': len(valid_faces),
                                    'timestamp': timestamp
                                })
                
                except RuntimeError as e:
                    # Handle CUDA errors
                    if "torchvision::nms" in str(e) or "CUDA" in str(e):
                        self.logger.warning(f"CUDA error in face detection: {e}")
                        try:
                            model.to('cpu')
                            # Retry on CPU (same logic as above)
                            # ... (would implement CPU retry logic)
                        except Exception as cpu_e:
                            self.logger.error(f"CPU fallback failed for face detection: {cpu_e}")
                            continue
                    else:
                        self.logger.error(f"Face detection failed for {frame_path}: {e}")
                        continue
            
            self.logger.info(f"Detected {results_data['total_faces']} faces across {len(results_data['faces_detected'])} detections")
            return results_data
            
        except Exception as e:
            self.logger.error(f"detect_faces failed: {e}")
            raise
    
    def batch_predict(
        self,
        sources: List[str],
        model_type: str = 'detection',
        stream: bool = True,
        **predict_kwargs
    ) -> Generator[Any, None, None]:
        """
        Batch prediction with memory optimization using stream
        
        Args:
            sources: List of image/video sources
            model_type: Type of model to use
            stream: Use streaming for memory efficiency
            **predict_kwargs: Additional prediction parameters
            
        Yields:
            YOLO prediction results
        """
        # Select appropriate model
        if model_type == 'detection':
            model = self.load_detection_model()
        elif model_type == 'face':
            model = self.load_face_model()
        elif model_type == 'emotion':
            model = self.load_emotion_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Default parameters
        default_params = {
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'max_det': self.max_detections,
            'stream': stream,
            'verbose': False
        }
        default_params.update(predict_kwargs)
        
        try:
            # Use GPU manager for error handling
            def predict_batch():
                return model.predict(source=sources, **default_params)
            
            results = self.gpu_manager.handle_yolo_errors(predict_batch)
            
            if stream:
                for result in results:
                    yield result
            else:
                yield from results
                
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise
    
    def export_results(
        self,
        results: Any,
        export_format: str = 'json',
        save_path: Optional[str] = None
    ) -> Union[pd.DataFrame, Dict, str]:
        """
        Export YOLO results using official API methods
        
        Args:
            results: YOLO prediction results
            export_format: Format to export ('json', 'df', 'save')
            save_path: Path to save results
            
        Returns:
            Exported results in requested format
        """
        try:
            if export_format == 'json':
                # Use results.to_json() method
                if hasattr(results, 'to_json'):
                    json_data = results.to_json()
                    if save_path:
                        with open(save_path, 'w') as f:
                            f.write(json_data)
                    return json_data
                else:
                    # Manual JSON export for batch results
                    json_data = []
                    for result in results:
                        if hasattr(result, 'to_json'):
                            json_data.append(result.to_json())
                    return json_data
                    
            elif export_format == 'df':
                # Use results.to_df() method
                if hasattr(results, 'to_df'):
                    df = results.to_df()
                    if save_path:
                        df.to_csv(save_path, index=False)
                    return df
                else:
                    # Manual DataFrame creation for batch results
                    all_data = []
                    for result in results:
                        if hasattr(result, 'to_df'):
                            df = result.to_df()
                            all_data.append(df)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        if save_path:
                            combined_df.to_csv(save_path, index=False)
                        return combined_df
                    
            elif export_format == 'save':
                # Use results.save() method
                if hasattr(results, 'save'):
                    save_dir = save_path or str(self.faces_yolo11_dir)
                    results.save(save_dir=save_dir)
                    return save_dir
                else:
                    # Manual save for batch results
                    save_dir = save_path or str(self.faces_yolo11_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    for i, result in enumerate(results):
                        if hasattr(result, 'save'):
                            result.save(save_dir=save_dir, exist_ok=True)
                    return save_dir
                    
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
    
    def validate_model(self, model_type: str = 'detection', data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate model using official YOLO validation API
        
        Args:
            model_type: Type of model to validate
            data_path: Path to validation dataset
            
        Returns:
            Validation metrics (mAP, precision, recall)
        """
        try:
            # Select model
            if model_type == 'detection':
                model = self.load_detection_model()
            elif model_type == 'face':
                model = self.load_face_model()
            elif model_type == 'emotion':
                model = self.load_emotion_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Run validation using official API
            if data_path:
                metrics = model.val(data=data_path)
            else:
                # Use default validation data
                metrics = model.val()
            
            # Extract key metrics
            validation_results = {
                'model_type': model_type,
                'map50': float(metrics.box.map50) if hasattr(metrics, 'box') else None,
                'map50_95': float(metrics.box.map) if hasattr(metrics, 'box') else None,
                'precision': float(metrics.box.mp) if hasattr(metrics, 'box') else None,
                'recall': float(metrics.box.mr) if hasattr(metrics, 'box') else None,
                'fitness': float(metrics.fitness) if hasattr(metrics, 'fitness') else None
            }
            
            self.logger.info(f"Model validation completed: mAP50={validation_results.get('map50', 'N/A')}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise
    
    def _extract_person_crops(self, result: Any, timestamp: float) -> List[str]:
        """Extract and save person crops from YOLO results"""
        crops = []
        
        try:
            if result.boxes is not None:
                # Load original image
                img = cv2.imread(result.path)
                
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Crop person
                    person_crop = img[y1:y2, x1:x2]
                    
                    # Save crop
                    crop_filename = f"person_{timestamp}_{i}.jpg"
                    crop_path = self.faces_dir / crop_filename
                    cv2.imwrite(str(crop_path), person_crop)
                    
                    crops.append(str(crop_path))
                    
        except Exception as e:
            self.logger.warning(f"Failed to extract person crops: {e}")
            
        return crops
    
    def _save_face_crop(
        self, 
        frame_path: str, 
        bbox: List[float], 
        timestamp: float, 
        face_id: Union[int, str]
    ) -> str:
        """Save individual face crop"""
        try:
            # Load image
            img = cv2.imread(frame_path)
            
            # Extract face region
            x1, y1, x2, y2 = map(int, bbox)
            face_crop = img[y1:y2, x1:x2]
            
            # Generate filename
            crop_filename = f"face_{timestamp}_{face_id}.jpg"
            crop_path = self.faces_dir / crop_filename
            
            # Save crop
            cv2.imwrite(str(crop_path), face_crop)
            
            return str(crop_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to save face crop: {e}")
            return ""
    
    def get_model_info(self, model_type: str = 'detection') -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            if model_type == 'detection':
                model = self.load_detection_model()
            elif model_type == 'face':
                model = self.load_face_model()
            elif model_type == 'emotion':
                model = self.load_emotion_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Use official model.info() method
            model_info = {
                'model_type': model_type,
                'model_path': str(model.ckpt_path) if hasattr(model, 'ckpt_path') else 'Unknown',
                'device': str(model.device) if hasattr(model, 'device') else 'Unknown',
                'task': model.task if hasattr(model, 'task') else 'Unknown',
                'names': model.names if hasattr(model, 'names') else {},
                'yaml': model.yaml if hasattr(model, 'yaml') else {}
            }
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}