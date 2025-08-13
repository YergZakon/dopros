import logging
import logging.handlers
from pathlib import Path
import yaml
import os

def setup_logger(config_path: str = "config.yaml") -> logging.Logger:
    """
    Setup centralized logging for DOPROS MVP 2.0
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured logger instance
    """
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Get logging configuration
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'dopros.log')
    max_size_mb = log_config.get('max_size_mb', 100)
    backup_count = log_config.get('backup_count', 5)
    
    # Create logger
    logger = logging.getLogger('dopros')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Name of the module/logger
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'dopros.{name}')
    return logging.getLogger('dopros')