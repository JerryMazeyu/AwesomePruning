# Import datasets from different sources
from .toys import DATASET_MAP as TOY_DATASETS
from .hf_data import DATASET_MAP as HF_DATASETS
from .toys import get_dataset as get_toy_dataset
from .hf_data import get_dataset as get_hf_dataset

# Create a unified dataset registry
DATASET_REGISTRY = {
    # Toy datasets (CV datasets)
    **{name: ('toy', name) for name in TOY_DATASETS.keys()},
    # HuggingFace datasets (NLP datasets)
    **{name: ('hf', name) for name in HF_DATASETS.keys()}
}

def list_datasets(dataset_type=None):
    """List all available datasets or datasets of a specific type
    
    Args:
        dataset_type (str, optional): Dataset type ('toy' or 'hf'). If None, list all datasets.
    
    Returns:
        list: List of dataset names
    """
    if dataset_type is None:
        return list(DATASET_REGISTRY.keys())
    
    return [name for name, (type_, _) in DATASET_REGISTRY.items() if type_ == dataset_type]

def get_dataset(name, **kwargs):
    """Get a dataset by name, automatically selecting the appropriate source
    
    Args:
        name (str): Dataset name
        **kwargs: Additional arguments for the dataset
    
    Returns:
        BaseDataSet: The requested dataset
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {list_datasets()}")
    
    dataset_type, dataset_name = DATASET_REGISTRY[name]
    
    if dataset_type == 'toy':
        return get_toy_dataset(dataset_name, **kwargs)
    elif dataset_type == 'hf':
        return get_hf_dataset(dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# Export the main interface
__all__ = ['list_datasets', 'get_dataset']