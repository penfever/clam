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
- Unified dataset preparation with intelligent caching and checking

The DatasetPreparer class provides a unified interface for dataset preparation that:
- Checks if datasets are already prepared before re-downloading
- Separates download and organization steps with intelligent caching
- Provides consistent logging and error handling
- Integrates with the dataset registry for tracking

Usage examples:
    # Simple CIFAR preparation using convenience function
    train_paths, train_labels, test_paths, test_labels, class_names = prepare_cifar_dataset("cifar10")
    
    # Custom dataset preparation using the DatasetPreparer
    resource_manager = get_resource_manager()
    success = resource_manager.dataset_preparer.prepare_dataset(
        dataset_id="my_dataset",
        dataset_type="vision",
        check_function=my_check_function,
        download_function=my_download_function,
        organize_function=my_organize_function
    )
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
        self.dataset_preparer = DatasetPreparer(self.path_resolver)
        
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


class DatasetPreparer:
    """Unified dataset preparation with intelligent caching and checking."""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
    
    def prepare_dataset(
        self,
        dataset_id: str,
        dataset_type: str,
        check_function: callable,
        download_function: callable,
        organize_function: callable = None,
        force_redownload: bool = False,
        force_reorganize: bool = False,
        min_samples: int = 100
    ) -> bool:
        """
        Unified dataset preparation with intelligent caching.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_type: Type of dataset (e.g., 'vision', 'audio', 'tabular')
            check_function: Function that returns True if dataset is ready
            download_function: Function that downloads/extracts the dataset
            organize_function: Optional function that organizes downloaded data
            force_redownload: Force re-download even if dataset exists
            force_reorganize: Force re-organization even if already organized
            min_samples: Minimum number of samples expected in prepared dataset
            
        Returns:
            True if dataset is ready, False otherwise
        """
        logger.info(f"Preparing {dataset_type} dataset: {dataset_id}")
        
        # Get dataset workspace
        dataset_dir = self.path_resolver.get_dataset_dir(dataset_id)
        
        # Check if dataset is already prepared
        if not force_redownload and not force_reorganize:
            try:
                if check_function(dataset_dir):
                    logger.info(f"Dataset {dataset_id} already prepared, skipping preparation")
                    return True
            except Exception as e:
                logger.debug(f"Dataset check failed for {dataset_id}: {e}")
        
        # Check if we need to download
        download_needed = force_redownload
        if not download_needed:
            try:
                # Check if raw data exists (but might need organization)
                download_needed = not self._check_raw_data_exists(dataset_dir, dataset_id)
            except Exception as e:
                logger.debug(f"Raw data check failed for {dataset_id}: {e}")
                download_needed = True
        
        # Download if needed
        if download_needed:
            try:
                logger.info(f"Downloading dataset {dataset_id}...")
                download_function(dataset_dir)
                logger.info(f"Download completed for {dataset_id}")
            except Exception as e:
                logger.error(f"Download failed for {dataset_id}: {e}")
                return False
        else:
            logger.info(f"Raw data for {dataset_id} already exists, skipping download")
        
        # Organize if needed and function provided
        if organize_function:
            organize_needed = force_reorganize
            if not organize_needed:
                try:
                    organize_needed = not check_function(dataset_dir)
                except Exception as e:
                    logger.debug(f"Organization check failed for {dataset_id}: {e}")
                    organize_needed = True
            
            if organize_needed:
                try:
                    logger.info(f"Organizing dataset {dataset_id}...")
                    organize_function(dataset_dir)
                    logger.info(f"Organization completed for {dataset_id}")
                except Exception as e:
                    logger.error(f"Organization failed for {dataset_id}: {e}")
                    return False
            else:
                logger.info(f"Dataset {dataset_id} already organized, skipping organization")
        
        # Final check
        try:
            if check_function(dataset_dir):
                logger.info(f"Dataset {dataset_id} successfully prepared")
                
                # Register dataset in registry
                metadata = {
                    'dataset_id': dataset_id,
                    'dataset_type': dataset_type,
                    'status': 'prepared',
                    'min_samples': min_samples,
                    'workspace_dir': str(dataset_dir)
                }
                
                dataset_registry = DatasetRegistry(self.path_resolver)
                dataset_registry.register_dataset(dataset_id, metadata)
                
                return True
            else:
                logger.error(f"Dataset {dataset_id} preparation verification failed")
                return False
        except Exception as e:
            logger.error(f"Final check failed for {dataset_id}: {e}")
            return False
    
    def _check_raw_data_exists(self, dataset_dir: Path, dataset_id: str) -> bool:
        """Check if raw downloaded data exists (before organization)."""
        # Look for common indicators of downloaded data
        indicators = [
            # Zip files
            list(dataset_dir.glob("*.zip")),
            # Tar files
            list(dataset_dir.glob("*.tar*")),
            # Extracted directories (excluding organized structure)
            [d for d in dataset_dir.iterdir() 
             if d.is_dir() and d.name not in ['images', 'audio', 'train', 'test', 'val', 'validation', '__pycache__']],
            # Raw data files
            list(dataset_dir.glob("*.csv")),
            list(dataset_dir.glob("*.json")),
            list(dataset_dir.glob("*.txt")),
        ]
        
        # If any indicator exists, consider raw data present
        return any(indicator for indicator in indicators)
    
    def prepare_cifar_dataset(self, dataset_type: str = "cifar10", force_redownload: bool = False) -> tuple:
        """
        Prepare CIFAR-10 or CIFAR-100 dataset using unified management.
        
        Args:
            dataset_type: "cifar10" or "cifar100"
            force_redownload: Force re-download even if dataset exists
            
        Returns:
            Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
        """
        from pathlib import Path
        import torchvision
        import torchvision.transforms as transforms
        
        if dataset_type == "cifar10":
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            min_images = 1000
            dataset_class = torchvision.datasets.CIFAR10
        elif dataset_type == "cifar100":
            class_names = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                'worm'
            ]
            min_images = 5000
            dataset_class = torchvision.datasets.CIFAR100
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        def check_cifar_prepared(dataset_dir: Path) -> bool:
            """Check if CIFAR dataset is properly prepared."""
            images_dir = dataset_dir / "images"
            if not images_dir.exists():
                return False
            
            # Check that we have train and test directories with images
            train_count = len(list(images_dir.glob("train/*/*.png")))
            test_count = len(list(images_dir.glob("test/*/*.png")))
            return train_count > min_images and test_count > 0
        
        def download_and_organize_cifar(dataset_dir: Path):
            """Download and organize CIFAR dataset in one step."""
            # Download dataset using torchvision
            transform = transforms.Compose([transforms.ToTensor()])
            
            train_dataset = dataset_class(
                root=str(dataset_dir), train=True, download=True, transform=transform
            )
            test_dataset = dataset_class(
                root=str(dataset_dir), train=False, download=True, transform=transform
            )
            
            # Organize into ImageNet-style structure
            images_dir = dataset_dir / "images"
            
            # Create directory structure
            for split in ['train', 'test']:
                for class_name in class_names:
                    (images_dir / split / class_name).mkdir(parents=True, exist_ok=True)
            
            # Convert and save images
            def save_cifar_images(dataset, base_dir: Path, class_names: list, split: str) -> tuple:
                """Save CIFAR images to disk in ImageNet-style structure."""
                paths = []
                labels = []
                
                logger.info(f"Saving {split} images...")
                
                for idx, (image_tensor, label) in enumerate(dataset):
                    if idx % 10000 == 0:
                        logger.info(f"Processed {idx}/{len(dataset)} {split} images")
                    
                    # Convert tensor to PIL Image
                    image = transforms.ToPILImage()(image_tensor)
                    
                    # Save image
                    class_name = class_names[label]
                    image_path = base_dir / class_name / f"{idx:05d}.png"
                    image.save(image_path)
                    
                    paths.append(str(image_path))
                    labels.append(label)
                
                return paths, labels
            
            train_paths, train_labels = save_cifar_images(
                train_dataset, images_dir / 'train', class_names, 'train'
            )
            test_paths, test_labels = save_cifar_images(
                test_dataset, images_dir / 'test', class_names, 'test'
            )
            
            logger.info(f"{dataset_type.upper()} prepared: {len(train_paths)} train, {len(test_paths)} test images")
        
        # Use unified preparation (combining download and organize into one step)
        success = self.prepare_dataset(
            dataset_id=dataset_type,
            dataset_type="vision",
            check_function=check_cifar_prepared,
            download_function=download_and_organize_cifar,
            organize_function=None,  # Already handled in download step
            force_redownload=force_redownload,
            min_samples=min_images
        )
        
        if not success:
            raise RuntimeError(f"Failed to prepare {dataset_type} dataset")
        
        # Load and return the prepared data
        def load_existing_cifar(images_dir: Path, class_names: list) -> tuple:
            """Load existing CIFAR directory structure."""
            train_paths, train_labels = [], []
            test_paths, test_labels = [], []
            
            for split, (paths_list, labels_list) in [('train', (train_paths, train_labels)), 
                                                    ('test', (test_paths, test_labels))]:
                for label, class_name in enumerate(class_names):
                    class_dir = images_dir / split / class_name
                    if class_dir.exists():
                        for img_path in sorted(class_dir.glob("*.png")):
                            paths_list.append(str(img_path))
                            labels_list.append(label)
            
            return train_paths, train_labels, test_paths, test_labels, class_names
        
        dataset_dir = self.path_resolver.get_dataset_dir(dataset_type)
        return load_existing_cifar(dataset_dir / "images", class_names)


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


# Convenience functions for common operations
def prepare_cifar_dataset(dataset_type: str = "cifar10", force_redownload: bool = False) -> tuple:
    """
    Convenience function to prepare CIFAR datasets using unified management.
    
    Args:
        dataset_type: "cifar10" or "cifar100"
        force_redownload: Force re-download even if dataset exists
        
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels, class_names)
    """
    resource_manager = get_resource_manager()
    return resource_manager.dataset_preparer.prepare_cifar_dataset(dataset_type, force_redownload)