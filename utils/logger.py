# utils/logger.py

import logging
import os

def setup_logger(log_path):
    """Set up the logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handler
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    
    # Create formatter and add to handler
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
