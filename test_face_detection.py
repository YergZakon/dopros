#!/usr/bin/env python3
"""
Быстрый тест детекции лиц в видео
"""

import cv2
import os
import numpy as np
from pathlib import Path

def test_opencv_face_detection(video_path, max_frames=10):
    """Тест детекции лиц с помощью OpenCV"""
    print(f"🔍 Тестируем детекцию лиц в {video_path}")
    
    # Загружаем Haar каскад для лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Не удалось открыть видео")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"📊 Видео: {total_frames} кадров, {fps:.1f} fps, {duration:.1f} сек")
    
    faces_detected = 0
    frames_tested = 0
    
    # Тестируем каждый 100-й кадр для быстроты
    frame_step = max(1, total_frames // max_frames)
    
    for frame_num in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        frames_tested += 1
        
        # Конвертируем в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детекция лиц
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            faces_detected += 1
            print(f"  ✅ Кадр {frame_num}: найдено {len(faces)} лиц")
            
            # Сохраняем первый кадр с лицом для проверки
            if faces_detected == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imwrite('test_face_detection.jpg', frame)
                print(f"  💾 Сохранен кадр с лицом: test_face_detection.jpg")
        
        if frames_tested >= max_frames:
            break
    
    cap.release()
    
    detection_rate = (faces_detected / frames_tested) * 100 if frames_tested > 0 else 0
    
    print(f"\n📈 Результаты:")
    print(f"  • Протестировано кадров: {frames_tested}")
    print(f"  • Кадров с лицами: {faces_detected}")
    print(f"  • Процент детекции: {detection_rate:.1f}%")
    
    return faces_detected > 0

def test_video_audio(video_path):
    """Проверяем наличие аудио в видео"""
    print(f"\n🎵 Проверяем аудио в {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Проверяем аудио через ffprobe
    import subprocess
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'stream=codec_type', '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True)
        
        streams = result.stdout.strip().split('\n')
        has_video = 'video' in streams
        has_audio = 'audio' in streams
        
        print(f"  • Видео поток: {'✅' if has_video else '❌'}")
        print(f"  • Аудио поток: {'✅' if has_audio else '❌'}")
        
        return has_audio
        
    except Exception as e:
        print(f"  ❌ Ошибка проверки аудио: {e}")
        return False

if __name__ == "__main__":
    video_path = "input_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Видео файл {video_path} не найден")
        exit(1)
    
    print("🚀 Запуск диагностики видео...")
    
    # Тест детекции лиц
    faces_ok = test_opencv_face_detection(video_path)
    
    # Тест аудио
    audio_ok = test_video_audio(video_path)
    
    print(f"\n🎯 Итоговая диагностика:")
    print(f"  • Лица: {'✅ обнаружены' if faces_ok else '❌ не найдены'}")
    print(f"  • Аудио: {'✅ присутствует' if audio_ok else '❌ отсутствует'}")
    
    if faces_ok and audio_ok:
        print("🎉 Видео подходит для анализа эмоций!")
    else:
        print("⚠️ Возможны проблемы с анализом")