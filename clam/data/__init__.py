"""
Data loading, processing, and preparation utilities.
"""

from .dataset import (
    load_dataset,
    load_datasets,
    analyze_dataset,
    create_llm_dataset,
    list_available_datasets,
    get_dataset_info
)

from .embeddings import (
    get_tabpfn_embeddings,
    prepare_tabpfn_embeddings_for_prefix
)

from .csv_utils import (
    is_csv_dataset,
    find_csv_file,
    load_csv_dataset,
    load_dataset_with_metadata,
    find_csv_with_fallbacks
)

from .evaluation_utils import (
    load_datasets_for_evaluation,
    preprocess_datasets_for_evaluation,
    validate_dataset_for_evaluation
)

__all__ = [
    "load_dataset",
    "load_datasets",
    "analyze_dataset",
    "create_llm_dataset",
    "get_tabpfn_embeddings",
    "prepare_tabpfn_embeddings_for_prefix",
    "list_available_datasets",
    "get_dataset_info",
    "is_csv_dataset",
    "find_csv_file",
    "load_csv_dataset",
    "load_dataset_with_metadata",
    "find_csv_with_fallbacks",
    "load_datasets_for_evaluation",
    "preprocess_datasets_for_evaluation",
    "validate_dataset_for_evaluation"
]