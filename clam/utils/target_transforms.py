"""
Target transformation utilities for handling extreme outliers in regression datasets.

This module provides automatic target transformation for regression datasets with
extreme value ranges that negatively impact visualization and VLM reasoning.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Union, Tuple
from sklearn.preprocessing import PowerTransformer, RobustScaler
import warnings

logger = logging.getLogger(__name__)


class TargetTransformer:
    """
    Handles target variable transformations for regression datasets with extreme ranges.
    
    Automatically detects when transformation is needed and applies appropriate
    methods like log transformation, power transformation, or robust scaling.
    """
    
    def __init__(self, method: str = "auto", range_threshold: float = 1000.0, 
                 ratio_threshold: float = 100.0):
        """
        Initialize target transformer.
        
        Args:
            method: Transformation method ("auto", "log1p", "power", "robust", "none")
            range_threshold: Minimum range (max-min) to trigger transformation
            ratio_threshold: Minimum ratio (max/min) to trigger transformation
        """
        self.method = method
        self.range_threshold = range_threshold
        self.ratio_threshold = ratio_threshold
        
        # State variables
        self.is_fitted = False
        self.needs_transform = False
        self.transform_method = None
        self.transformer = None
        self.original_stats = None
        self.transformed_stats = None
        
    def should_transform(self, y: np.ndarray) -> Tuple[bool, str]:
        """
        Determine if target transformation is needed based on range and distribution.
        
        Args:
            y: Target values
            
        Returns:
            (needs_transform, reason)
        """
        y_clean = y[~np.isnan(y)]
        
        if len(y_clean) == 0:
            return False, "no_valid_data"
        
        y_min, y_max = np.min(y_clean), np.max(y_clean)
        target_range = y_max - y_min
        
        # Avoid division by zero
        y_min_abs = max(abs(y_min), 1e-8)
        target_ratio = y_max / y_min_abs
        
        # Check if transformation criteria are met
        range_condition = target_range >= self.range_threshold
        ratio_condition = target_ratio >= self.ratio_threshold
        
        if range_condition and ratio_condition:
            reason = f"range={target_range:.2f}>={self.range_threshold}, ratio={target_ratio:.2f}>={self.ratio_threshold}"
            return True, reason
        
        reason = f"range={target_range:.2f}<{self.range_threshold} or ratio={target_ratio:.2f}<{self.ratio_threshold}"
        return False, reason
    
    def _choose_transform_method(self, y: np.ndarray) -> str:
        """
        Automatically choose the best transformation method based on data characteristics.
        
        Args:
            y: Target values
            
        Returns:
            Transform method name
        """
        y_clean = y[~np.isnan(y)]
        
        # Check if all values are positive (required for log transform)
        all_positive = np.all(y_clean > 0)
        has_zeros = np.any(y_clean == 0)
        
        # Check skewness to determine if log transform is appropriate
        from scipy import stats
        try:
            skewness = stats.skew(y_clean)
            high_skew = abs(skewness) > 2.0
        except:
            high_skew = False
        
        # Decision logic
        if (all_positive or has_zeros) and high_skew:
            # log1p handles zeros and positive values well for skewed data
            return "log1p"
        elif all_positive and high_skew:
            # Pure log transform for positive-only data
            return "log"
        else:
            # Power transform (Yeo-Johnson) handles negative values and finds optimal lambda
            return "power"
    
    def fit(self, y: np.ndarray) -> 'TargetTransformer':
        """
        Fit the transformer to the target data.
        
        Args:
            y: Target values to fit transformer on
            
        Returns:
            Self for method chaining
        """
        y = np.asarray(y)
        
        # Store original statistics
        y_clean = y[~np.isnan(y)]
        self.original_stats = {
            'min': np.min(y_clean),
            'max': np.max(y_clean),
            'mean': np.mean(y_clean),
            'std': np.std(y_clean),
            'median': np.median(y_clean),
            'q25': np.percentile(y_clean, 25),
            'q75': np.percentile(y_clean, 75),
            'range': np.max(y_clean) - np.min(y_clean),
            'count': len(y_clean)
        }
        
        # Determine if transformation is needed
        self.needs_transform, reason = self.should_transform(y)
        
        if not self.needs_transform:
            logger.info(f"Target transformation not needed: {reason}")
            self.transform_method = "none"
            self.is_fitted = True
            return self
        
        # Choose transformation method
        if self.method == "auto":
            self.transform_method = self._choose_transform_method(y)
        else:
            self.transform_method = self.method
        
        logger.info(f"Target transformation needed: {reason}")
        logger.info(f"Using transformation method: {self.transform_method}")
        
        # Fit the appropriate transformer
        try:
            if self.transform_method == "log1p":
                # log1p: log(1 + x) - handles zeros, requires x >= -1
                if np.any(y_clean < -1):
                    logger.warning("Some values < -1, shifting data for log1p transform")
                    self.shift = -np.min(y_clean) + 1
                else:
                    self.shift = 0
                self.transformer = None  # log1p doesn't need sklearn transformer
                
            elif self.transform_method == "log":
                # Pure log transform - requires positive values
                if np.any(y_clean <= 0):
                    logger.warning("Non-positive values found, adding shift for log transform")
                    self.shift = -np.min(y_clean) + 1e-8
                else:
                    self.shift = 0
                self.transformer = None
                
            elif self.transform_method == "power":
                # Yeo-Johnson power transform - handles negative values
                self.transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                self.transformer.fit(y_clean.reshape(-1, 1))
                self.shift = 0
                
            elif self.transform_method == "robust":
                # Robust scaling using median and IQR
                self.transformer = RobustScaler()
                self.transformer.fit(y_clean.reshape(-1, 1))
                self.shift = 0
                
            else:
                raise ValueError(f"Unknown transform method: {self.transform_method}")
            
            # Store transformed statistics for reference
            y_transformed = self.transform(y)
            y_transformed_clean = y_transformed[~np.isnan(y_transformed)]
            self.transformed_stats = {
                'min': np.min(y_transformed_clean),
                'max': np.max(y_transformed_clean),
                'mean': np.mean(y_transformed_clean),
                'std': np.std(y_transformed_clean),
                'median': np.median(y_transformed_clean),
                'range': np.max(y_transformed_clean) - np.min(y_transformed_clean)
            }
            
            logger.info(f"Transform fit successfully. Original range: {self.original_stats['range']:.2f}, "
                       f"Transformed range: {self.transformed_stats['range']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to fit transformer: {e}")
            # Fallback to no transformation
            self.needs_transform = False
            self.transform_method = "none"
        
        self.is_fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform target values using the fitted transformer.
        
        Args:
            y: Target values to transform
            
        Returns:
            Transformed target values
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        if not self.needs_transform:
            return y.copy()
        
        y = np.asarray(y)
        y_transformed = y.copy()
        
        # Handle NaN values
        valid_mask = ~np.isnan(y)
        y_valid = y[valid_mask]
        
        try:
            if self.transform_method == "log1p":
                y_valid_shifted = y_valid + self.shift
                y_transformed[valid_mask] = np.log1p(y_valid_shifted)
                
            elif self.transform_method == "log":
                y_valid_shifted = y_valid + self.shift
                y_transformed[valid_mask] = np.log(y_valid_shifted)
                
            elif self.transform_method in ["power", "robust"]:
                y_valid_transformed = self.transformer.transform(y_valid.reshape(-1, 1)).flatten()
                y_transformed[valid_mask] = y_valid_transformed
                
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            # Return original values on failure
            return y.copy()
        
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform values back to original scale.
        
        Args:
            y_transformed: Transformed target values
            
        Returns:
            Values in original scale
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        
        if not self.needs_transform:
            return y_transformed.copy()
        
        y_transformed = np.asarray(y_transformed)
        y_original = y_transformed.copy()
        
        # Handle NaN values
        valid_mask = ~np.isnan(y_transformed)
        y_valid = y_transformed[valid_mask]
        
        try:
            if self.transform_method == "log1p":
                y_original[valid_mask] = np.expm1(y_valid) - self.shift
                
            elif self.transform_method == "log":
                y_original[valid_mask] = np.exp(y_valid) - self.shift
                
            elif self.transform_method in ["power", "robust"]:
                y_valid_original = self.transformer.inverse_transform(y_valid.reshape(-1, 1)).flatten()
                y_original[valid_mask] = y_valid_original
                
        except Exception as e:
            logger.error(f"Inverse transform failed: {e}")
            # Return transformed values on failure (better than crashing)
            return y_transformed.copy()
        
        return y_original
    
    def get_transform_info(self) -> Dict[str, Any]:
        """
        Get information about the transformation for logging/debugging.
        
        Returns:
            Dictionary with transformation details
        """
        info = {
            'is_fitted': self.is_fitted,
            'needs_transform': self.needs_transform,
            'transform_method': self.transform_method,
            'range_threshold': self.range_threshold,
            'ratio_threshold': self.ratio_threshold,
        }
        
        if self.original_stats:
            info['original_stats'] = self.original_stats.copy()
        
        if self.transformed_stats:
            info['transformed_stats'] = self.transformed_stats.copy()
            
        if hasattr(self, 'shift'):
            info['shift'] = self.shift
        
        return info


def detect_extreme_targets(y: np.ndarray, range_threshold: float = 1000.0, 
                          ratio_threshold: float = 100.0) -> Dict[str, Any]:
    """
    Detect if target values have extreme ranges that would benefit from transformation.
    
    Args:
        y: Target values
        range_threshold: Minimum range (max-min) to be considered extreme
        ratio_threshold: Minimum ratio (max/min) to be considered extreme
        
    Returns:
        Dictionary with detection results and statistics
    """
    y_clean = y[~np.isnan(y)]
    
    if len(y_clean) == 0:
        return {'is_extreme': False, 'reason': 'no_valid_data'}
    
    y_min, y_max = np.min(y_clean), np.max(y_clean)
    target_range = y_max - y_min
    y_min_abs = max(abs(y_min), 1e-8)
    target_ratio = y_max / y_min_abs
    
    is_extreme = target_range >= range_threshold and target_ratio >= ratio_threshold
    
    return {
        'is_extreme': is_extreme,
        'target_range': target_range,
        'target_ratio': target_ratio,
        'range_threshold': range_threshold,
        'ratio_threshold': ratio_threshold,
        'reason': f"range={target_range:.2f}, ratio={target_ratio:.2f}",
        'stats': {
            'min': y_min,
            'max': y_max,
            'mean': np.mean(y_clean),
            'std': np.std(y_clean),
            'median': np.median(y_clean),
            'count': len(y_clean)
        }
    }