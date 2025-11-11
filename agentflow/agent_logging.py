import logging
import sys
import os
from typing import Optional

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    log_level = level or os.environ.get('LOG_LEVEL', 'INFO')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return logger

def configure_agentflow_logging(level: str = "INFO"):
    """Configure root logging for AgentFlow."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def configure_logger(level: int = logging.INFO, name: str = "agentflow") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()  # clear existing handlers

    # log to stdout
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging
    return logger
