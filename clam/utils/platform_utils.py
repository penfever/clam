"""
Platform compatibility utilities for CLAM.

This module provides cross-platform compatibility functions for
device detection, model configuration, and platform-specific optimizations.
"""

import sys
import torch
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


def get_optimal_device(force_cpu: bool = False, prefer_mps: bool = False) -> str:
    """
    Get the optimal device for the current platform.
    
    Args:
        force_cpu: Force CPU usage regardless of available hardware
        prefer_mps: Prefer MPS on Mac if available (experimental)
        
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if force_cpu:
        return 'cpu'
    
    # Check for CUDA first (most common GPU setup)
    if torch.cuda.is_available():
        return 'cuda'
    
    # Check for MPS on Mac (Apple Silicon)
    if prefer_mps and sys.platform == "darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    
    # Default to CPU
    return 'cpu'


def get_platform_compatible_dtype(device: str = None) -> torch.dtype:
    """
    Get appropriate dtype for the current platform and device.
    
    Args:
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detect)
        
    Returns:
        Appropriate torch dtype
    """
    if device is None:
        device = get_optimal_device()
    
    # Mac compatibility: use float32 to avoid potential issues
    if sys.platform == "darwin":
        return torch.float32
    
    # Use bfloat16 for CUDA if available, float32 otherwise
    if device == 'cuda' and torch.cuda.is_available():
        return torch.bfloat16  # Changed from float16 to bfloat16 for better numerical stability
    
    return torch.float32


def configure_model_kwargs_for_platform(
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = True
) -> Dict[str, Any]:
    """
    Configure model loading kwargs based on platform capabilities.
    
    Args:
        device: Target device (auto-detected if None)
        torch_dtype: Target dtype (auto-selected if None)
        low_cpu_mem_usage: Use low CPU memory mode
        trust_remote_code: Allow loading models with custom code
        
    Returns:
        Dictionary of model loading kwargs
    """
    if device is None:
        device = get_optimal_device()
    
    if torch_dtype is None:
        torch_dtype = get_platform_compatible_dtype(device)
    
    kwargs = {
        'trust_remote_code': trust_remote_code,
        'low_cpu_mem_usage': low_cpu_mem_usage
    }
    
    # Device and dtype configuration
    if device == 'cuda' and torch.cuda.is_available():
        kwargs.update({
            'torch_dtype': torch_dtype,
            'device_map': 'auto'
        })
    elif device == 'cpu' or sys.platform == "darwin":
        # For CPU or Mac, use simpler configuration
        kwargs.update({
            'torch_dtype': torch_dtype,
            'device_map': 'cpu'
        })
    else:
        # MPS or other devices
        kwargs.update({
            'torch_dtype': torch_dtype
        })
    
    return kwargs


def is_mac_platform() -> bool:
    """Check if running on macOS."""
    return sys.platform == "darwin"


def log_platform_info(logger_instance: logging.Logger = None) -> Dict[str, Any]:
    """
    Log platform information and return as dictionary.
    
    Args:
        logger_instance: Logger to use (uses module logger if None)
        
    Returns:
        Dictionary with platform information
    """
    if logger_instance is None:
        logger_instance = logger
    
    optimal_device = get_optimal_device()
    dtype = get_platform_compatible_dtype(optimal_device)
    
    platform_info = {
        'platform': sys.platform,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) if sys.platform == "darwin" else False,
        'optimal_device': optimal_device,
        'recommended_dtype': str(dtype)
    }
    
    # More concise platform logging
    logger_instance.info(f"Platform: {platform_info['platform']}, PyTorch: {platform_info['torch_version']}, Device: {platform_info['optimal_device']}")
    
    return platform_info


def get_mac_compatible_dinov2_kwargs() -> Dict[str, Any]:
    """
    Get DINOV2-specific kwargs for Mac compatibility.
    
    Returns:
        Dictionary of kwargs for loading DINOV2 on Mac
    """
    if not is_mac_platform():
        return {}
    
    return {
        'device': 'cpu',
        'torch_dtype': torch.float32
    }


def suppress_platform_warnings():
    """
    Suppress common platform-specific warnings.
    
    This includes MPS warnings on Mac, CUDA warnings on CPU-only systems, etc.
    """
    import warnings
    
    # Suppress MPS warnings on Mac
    if is_mac_platform():
        warnings.filterwarnings("ignore", message=".*MPS.*")
        warnings.filterwarnings("ignore", message=".*Metal.*")
    
    # Suppress CUDA warnings when CUDA not available
    if not torch.cuda.is_available():
        warnings.filterwarnings("ignore", message=".*CUDA.*")
        warnings.filterwarnings("ignore", message=".*cuda.*")