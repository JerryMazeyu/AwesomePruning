from .base import BaseDataSet
from datasets import load_dataset
from config import CONF
import warnings
import os
warnings.filterwarnings("ignore")

# Define commonly used HuggingFace dataset mappings
DATASET_MAP = {
    # Text classification datasets
    'imdb': 'imdb',
    # 'sst2': 'glue/sst2',
    # 'mnli': 'glue/mnli',
    # Language modeling datasets
    'wikitext2': 'Salesforce/wikitext',
    # 'wikitext103': 'wikitext/wikitext-103-raw-v1',
    # 'c4': 'c4',
    # Question answering datasets
    # 'squad': 'squad',
    # 'squad_v2': 'squad_v2'
}

# Dataset parameter mappings
ARG_MAP = {
    'imdb': {'split': 'train'},
    # 'sst2': {'split': 'train'},
    # 'mnli': {'split': 'train'},
    'wikitext2': {'name': 'wikitext-2-raw-v1', 'split': 'train'},
    # 'wikitext103': {'split': 'train'},
    # 'c4': {'split': 'train'},
    # 'squad': {'split': 'train'},
    # 'squad_v2': {'split': 'train'}
}

# Dataset metadata
META_MAP = {
    'imdb': {
        'task_type': 'sequence_classification',
        'num_labels': 2,
        'description': "IMDB电影评论情感分类数据集"
    },
    'sst2': {
        'task_type': 'sequence_classification',
        'num_labels': 2,
        'description': "SST-2情感分类数据集"
    },
    'wikitext2': {
        'task_type': 'language_modeling',
        'description': "WikiText-2语言建模数据集"
    },
    'wikitext103': {
        'task_type': 'language_modeling',
        'description': "WikiText-103语言建模数据集"
    }
}

def get_cache_path(dataset_name: str) -> str:
    """Get the local cache path for the dataset

    Args:
        dataset_name (str): Dataset name

    Returns:
        str: Cache path
    """
    # Replace '/' in dataset name with '_' to create a valid file path
    safe_name = dataset_name.replace('/', '_')
    return os.path.join(CONF.data_root_path, 'hf_datasets', safe_name)

def dataset_list(name: str = None, **kwargs):
    """Get dataset list or a specific dataset

    Args:
        name (str, optional): Dataset name, returns all available datasets if None. Defaults to None.

    Returns:
        Union[list, Dataset]: Dataset list or the specified dataset
    """
    if not name:
        return list(DATASET_MAP.keys())
    else:
        if name not in DATASET_MAP:
            raise ValueError(f"Dataset {name} is not in the supported list. Supported datasets: {list(DATASET_MAP.keys())}")
        
        # Update parameters
        args = ARG_MAP[name].copy()
        args.update(kwargs)
        
        # Get cache path
        cache_dir = get_cache_path(DATASET_MAP[name])
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load dataset
        try:
            # Add cache_dir parameter to enable local caching
            dataset = load_dataset(
                DATASET_MAP[name],
                cache_dir=cache_dir,
                **args
            )
            
            if isinstance(dataset, dict):
                # Some datasets may return DatasetDict, we get the specified split
                split = args.get('split', 'train')
                dataset = dataset[split]
            
            print(f"Dataset cache path: {cache_dir}")
            return dataset
            
        except Exception as e:
            raise Exception(f"Error occurred while loading dataset {name}: {str(e)}")

def get_dataset(dt_name: str, **kwargs):
    """Get the specified dataset and its metadata

    Args:
        dt_name (str): Dataset name
        **kwargs: Other parameters

    Returns:
        BaseDataSet: Wrapped dataset
    """
    dt = dataset_list(dt_name, **kwargs)
    meta = META_MAP.get(dt_name, {})
    return BaseDataSet(dt, meta)

if __name__ == "__main__":
    # Test code
    print("Available dataset list:", dataset_list())
    
    # Test loading IMDB dataset
    print("\nLoading IMDB dataset...")
    imdb_dataset = get_dataset('imdb')
    print("IMDB dataset metadata:", imdb_dataset.meta)
    
    # Test loading WikiText-2 dataset
    print("\nLoading WikiText-2 dataset...")
    wiki_dataset = get_dataset('wikitext2')
    print("WikiText-2 dataset metadata:", wiki_dataset.meta) 