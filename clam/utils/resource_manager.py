"""
Robust resource management system for CLAM.

This module provides centralized, package-aware resource management that eliminates
fragile path-guessing logic and provides consistent resource organization across
different deployment environments.

Key features:
- Package-aware path resolution using importlib.resources
- Consistent dataset workspace isolation
- Environment variable configuration support
- Backward compatibility with existing code
- Centralized config and metadata management
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import importlib.resources as resources

logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """Configuration for CLAM resource management."""
    base_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    datasets_dir: Optional[str] = None
    configs_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> 'ResourceConfig':
        """Create config from environment variables."""
        return cls(
            base_dir=os.environ.get('CLAM_BASE_DIR'),
            cache_dir=os.environ.get('CLAM_CACHE_DIR'),
            datasets_dir=os.environ.get('CLAM_DATASETS_DIR'),
            configs_dir=os.environ.get('CLAM_CONFIGS_DIR'),
            temp_dir=os.environ.get('CLAM_TEMP_DIR'),
        )


@dataclass
class DatasetMetadata:
    """Metadata for dataset management and caching."""
    dataset_id: str
    name: Optional[str] = None
    task_type: Optional[str] = None
    feature_count: Optional[int] = None
    sample_count: Optional[int] = None
    class_count: Optional[int] = None
    openml_task_id: Optional[int] = None
    source: Optional[str] = None
    cached_at: Optional[str] = None
    file_paths: Optional[Dict[str, str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'task_type': self.task_type,
            'feature_count': self.feature_count,
            'sample_count': self.sample_count,
            'class_count': self.class_count,
            'openml_task_id': self.openml_task_id,
            'source': self.source,
            'cached_at': self.cached_at,
            'file_paths': self.file_paths or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create from dictionary."""
        return cls(**data)


class PathResolver:
    """Robust path resolution using package-aware methods."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self._base_dir = None
    
    def get_base_dir(self) -> Path:
        """Get base CLAM directory."""
        if self._base_dir is None:
            if self.config.base_dir:
                self._base_dir = Path(self.config.base_dir).expanduser().resolve()
            else:
                self._base_dir = Path.home() / '.clam'
            self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir
    
    def get_cache_dir(self) -> Path:
        """Get cache directory."""
        if self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir).expanduser().resolve()
        else:
            cache_dir = self.get_base_dir() / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_datasets_dir(self) -> Path:
        """Get datasets directory."""
        if self.config.datasets_dir:
            datasets_dir = Path(self.config.datasets_dir).expanduser().resolve()
        else:
            datasets_dir = self.get_base_dir() / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)
        return datasets_dir
    
    def get_configs_dir(self) -> Path:
        """Get configs directory."""
        if self.config.configs_dir:
            configs_dir = Path(self.config.configs_dir).expanduser().resolve()
        else:
            configs_dir = self.get_base_dir() / 'configs'
        configs_dir.mkdir(parents=True, exist_ok=True)
        return configs_dir
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory."""
        if self.config.temp_dir:
            temp_dir = Path(self.config.temp_dir).expanduser().resolve()
        else:
            temp_dir = self.get_base_dir() / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def get_dataset_dir(self, dataset_id: str) -> Path:
        """Get directory for a specific dataset."""
        dataset_dir = self.get_datasets_dir() / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir
    
    def get_embedding_dir(self, dataset_id: str) -> Path:
        """Get embeddings directory for a specific dataset."""
        embed_dir = self.get_dataset_dir(dataset_id) / 'embeddings'
        embed_dir.mkdir(parents=True, exist_ok=True)
        return embed_dir
    
    def get_dataset_cache_dir(self, dataset_id: str) -> Path:
        """Get cache directory for a specific dataset."""
        cache_dir = self.get_dataset_dir(dataset_id) / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_config_path(self, model_type: str, filename: str) -> Optional[Path]:
        """Get config file path with fallback to package resources."""
        # First try managed configs directory
        config_path = self.get_configs_dir() / model_type / filename
        if config_path.exists():
            return config_path
        
        # Try package resources for backward compatibility
        try:
            # Handle different config patterns
            if model_type == 'jolt':
                package_path = f'clam.examples.tabular.llm_baselines.jolt'
            elif model_type == 'tabllm':
                package_path = f'clam.examples.tabular.llm_baselines.tabllm_like'
            elif model_type == 'cc18_semantic':
                package_path = f'clam.data.cc18_semantic'
            else:
                return None
            
            # Try to get path from package resources
            try:
                with resources.path(package_path, filename) as path:
                    if path.exists():
                        return path
            except (ModuleNotFoundError, FileNotFoundError, AttributeError):
                # Fallback: try to find package directory manually
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent
                
                if model_type == 'jolt':
                    fallback_path = project_root / 'examples' / 'tabular' / 'llm_baselines' / 'jolt' / filename
                elif model_type == 'tabllm':
                    fallback_path = project_root / 'examples' / 'tabular' / 'llm_baselines' / 'tabllm_like' / filename
                elif model_type == 'cc18_semantic':
                    fallback_path = project_root / 'data' / 'cc18_semantic' / filename
                else:
                    return None
                
                if fallback_path.exists():
                    return fallback_path
                    
        except Exception as e:
            logger.debug(f"Error accessing package resources for {model_type}/{filename}: {e}")
        
        return None


class DatasetRegistry:
    """Registry for tracking dataset metadata and locations."""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
        self._registry = {}
        self._load_registry()
    
    def _get_registry_path(self) -> Path:
        """Get path to registry file."""
        return self.path_resolver.get_base_dir() / 'dataset_registry.json'
    
    def _load_registry(self):
        """Load registry from disk."""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    self._registry = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading dataset registry: {e}")
                self._registry = {}
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_path = self._get_registry_path()
        try:
            with open(registry_path, 'w') as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving dataset registry: {e}")
    
    def register_dataset(self, dataset_id: str, metadata: Dict[str, Any]):
        """Register a dataset with metadata."""
        import datetime
        self._registry[dataset_id] = {
            'metadata': metadata,
            'registered_at': datetime.datetime.now().isoformat(),
            'workspace_dir': str(self.path_resolver.get_dataset_dir(dataset_id))
        }
        self._save_registry()
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset information."""
        return self._registry.get(dataset_id)
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self._registry.keys())
    
    def unregister_dataset(self, dataset_id: str):
        """Unregister a dataset."""
        if dataset_id in self._registry:
            del self._registry[dataset_id]
            self._save_registry()


class CacheManager:
    """Manager for caching data with smart organization."""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
    
    def get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        import hashlib
        key_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_type: str, cache_key: str, extension: str = '') -> Path:
        """Get path for cached data."""
        cache_dir = self.path_resolver.get_cache_dir() / cache_type
        cache_dir.mkdir(exist_ok=True)
        filename = f"{cache_key}{extension}"
        return cache_dir / filename
    
    def cache_exists(self, cache_type: str, cache_key: str, extension: str = '') -> bool:
        """Check if cache file exists."""
        cache_path = self.get_cache_path(cache_type, cache_key, extension)
        return cache_path.exists()
    
    def save_to_cache(self, cache_type: str, cache_key: str, data: Any, extension: str = '') -> bool:
        """Save data to cache."""
        try:
            cache_path = self.get_cache_path(cache_type, cache_key, extension)
            
            if extension == '.json':
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif extension == '.npz':
                import numpy as np
                if isinstance(data, dict):
                    np.savez(cache_path, **data)
                else:
                    np.savez(cache_path, data=data)
            else:
                # Generic pickle fallback
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return True
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
            return False
    
    def load_from_cache(self, cache_type: str, cache_key: str, extension: str = '') -> Optional[Any]:
        """Load data from cache."""
        try:
            cache_path = self.get_cache_path(cache_type, cache_key, extension)
            if not cache_path.exists():
                return None
            
            if extension == '.json':
                with open(cache_path, 'r') as f:
                    return json.load(f)
            elif extension == '.npz':
                import numpy as np
                return np.load(cache_path, allow_pickle=True)
            else:
                # Generic pickle fallback
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None


class ConfigManager:
    """Manager for config file discovery and validation."""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
    
    def find_jolt_config(self, dataset_name: str) -> Optional[Path]:
        """Find JOLT config for dataset."""
        filename = f'jolt_config_{dataset_name}.json'
        return self.path_resolver.get_config_path('jolt', filename)
    
    def find_tabllm_template(self, dataset_name: str) -> Optional[Path]:
        """Find TabLLM template for dataset."""
        filename = f'templates_{dataset_name}.yaml'
        return self.path_resolver.get_config_path('tabllm', filename)
    
    def find_tabllm_notes(self, dataset_name: str) -> Optional[Path]:
        """Find TabLLM notes for dataset."""
        filename = f'notes_{dataset_name}.jsonl'
        # Notes are in a subdirectory
        notes_path = self.path_resolver.get_configs_dir() / 'tabllm' / 'notes' / filename
        if notes_path.exists():
            return notes_path
        
        # Fallback to package resources
        return self.path_resolver.get_config_path('tabllm', f'notes/{filename}')
    
    def find_cc18_semantic(self, dataset_name: str) -> Optional[Path]:
        """Find CC18 semantic metadata for dataset."""
        filename = f'{dataset_name}.json'
        return self.path_resolver.get_config_path('cc18_semantic', filename)
    
    def get_openml_task_mapping(self, model_type: str) -> Optional[Dict[str, int]]:
        """Get OpenML task mapping for a model type."""
        filename = 'openml_task_mapping.json'
        mapping_path = self.path_resolver.get_config_path(model_type, filename)
        
        if mapping_path and mapping_path.exists():
            try:
                with open(mapping_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading OpenML task mapping for {model_type}: {e}")
        
        return None


class ClamResourceManager:
    """Main resource manager for CLAM."""
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        if config is None:
            config = ResourceConfig.from_environment()
        
        self.config = config
        self.path_resolver = PathResolver(config)
        self.dataset_registry = DatasetRegistry(self.path_resolver)
        self.config_manager = ConfigManager(self.path_resolver)
        self.cache_manager = CacheManager(self.path_resolver)
        
        logger.debug(f"Initialized CLAM resource manager with base dir: {self.path_resolver.get_base_dir()}")
    
    def get_dataset_workspace(self, dataset_id: str) -> Path:
        """Get workspace directory for a dataset."""
        return self.path_resolver.get_dataset_dir(dataset_id)
    
    def find_csv_file(self, dataset_id: str, additional_search_dirs: Optional[List[str]] = None) -> Optional[Path]:
        """Find CSV file using robust search strategy."""
        search_dirs = []
        
        # Priority 1: Dataset's own directory
        dataset_dir = self.path_resolver.get_dataset_dir(dataset_id)
        search_dirs.append(dataset_dir)
        
        # Priority 2: Additional search directories provided by user
        if additional_search_dirs:
            search_dirs.extend([Path(d) for d in additional_search_dirs])
        
        # Priority 3: Common data directories
        search_dirs.extend([
            self.path_resolver.get_datasets_dir(),
            Path.cwd() / 'data',
            Path.cwd(),
        ])
        
        # Search for CSV files
        csv_patterns = [
            f'{dataset_id}.csv',
            f'dataset_{dataset_id}.csv',
            f'{dataset_id}_processed.csv',
            f'{dataset_id}_clean.csv',
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for pattern in csv_patterns:
                csv_path = search_dir / pattern
                if csv_path.exists():
                    logger.debug(f"Found CSV file: {csv_path}")
                    return csv_path
                
                # Also check subdirectories
                for subdir in search_dir.iterdir():
                    if subdir.is_dir():
                        csv_path = subdir / pattern
                        if csv_path.exists():
                            logger.debug(f"Found CSV file in subdirectory: {csv_path}")
                            return csv_path
        
        logger.debug(f"CSV file not found for dataset {dataset_id} in search directories")
        return None
    
    def validate_model_metadata(self, openml_task_id: int, model_type: str) -> Dict[str, Any]:
        """Validate metadata for a specific model type."""
        result = {
            'valid': True,
            'missing_files': [],
            'errors': [],
            'warnings': [],
            'dataset_name': None
        }
        
        # Get task mapping
        task_mapping = self.config_manager.get_openml_task_mapping(model_type)
        if not task_mapping:
            result['valid'] = False
            result['errors'].append(f"OpenML task mapping not found for {model_type}")
            return result
        
        # Find dataset name for this task ID
        dataset_name = None
        for name, task_id in task_mapping.items():
            if task_id == openml_task_id:
                dataset_name = name
                break
        
        if dataset_name is None:
            result['valid'] = False
            result['errors'].append(f"No {model_type} config found for OpenML task ID {openml_task_id}")
            return result
        
        result['dataset_name'] = dataset_name
        
        # Check model-specific files
        if model_type == 'jolt':
            config_path = self.config_manager.find_jolt_config(dataset_name)
            if not config_path:
                result['valid'] = False
                result['errors'].append(f"JOLT config file not found for {dataset_name}")
        
        elif model_type == 'tabllm':
            template_path = self.config_manager.find_tabllm_template(dataset_name)
            if not template_path:
                result['valid'] = False
                result['errors'].append(f"TabLLM template file not found for {dataset_name}")
            
            notes_path = self.config_manager.find_tabllm_notes(dataset_name)
            if not notes_path:
                result['warnings'].append(f"TabLLM notes file not found for {dataset_name}")
        
        return result


# Global resource manager instance
_resource_manager: Optional[ClamResourceManager] = None


def get_resource_manager(config: Optional[ResourceConfig] = None) -> ClamResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ClamResourceManager(config)
    return _resource_manager


def reset_resource_manager():
    """Reset global resource manager instance (mainly for testing)."""
    global _resource_manager
    _resource_manager = None