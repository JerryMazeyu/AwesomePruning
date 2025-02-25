from base import BaseDataSet
from datasets import load_dataset
from config import CONF
import warnings
import os
warnings.filterwarnings("ignore")

# 定义常用的HuggingFace数据集映射
DATASET_MAP = {
    # 文本分类数据集
    'imdb': 'imdb',
    'sst2': 'glue/sst2',
    'mnli': 'glue/mnli',
    # 语言建模数据集
    'wikitext2': 'wikitext/wikitext-2-raw-v1',
    'wikitext103': 'wikitext/wikitext-103-raw-v1',
    'c4': 'c4',
    # 问答数据集
    'squad': 'squad',
    'squad_v2': 'squad_v2'
}

# 数据集参数映射
ARG_MAP = {
    'imdb': {'split': 'train'},
    'sst2': {'split': 'train'},
    'mnli': {'split': 'train'},
    'wikitext2': {'split': 'train'},
    'wikitext103': {'split': 'train'},
    'c4': {'split': 'train'},
    'squad': {'split': 'train'},
    'squad_v2': {'split': 'train'}
}

# 数据集元信息
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
    """获取数据集的本地缓存路径

    Args:
        dataset_name (str): 数据集名称

    Returns:
        str: 缓存路径
    """
    # 将数据集名称中的'/'替换为'_'以创建有效的文件路径
    safe_name = dataset_name.replace('/', '_')
    return os.path.join(CONF.data_root_path, 'hf_datasets', safe_name)

def dataset_list(name: str = None, **kwargs):
    """获取数据集列表或指定数据集

    Args:
        name (str, optional): 数据集名称，如果为None则返回所有可用数据集列表. Defaults to None.

    Returns:
        Union[list, Dataset]: 数据集列表或指定的数据集
    """
    if not name:
        return list(DATASET_MAP.keys())
    else:
        if name not in DATASET_MAP:
            raise ValueError(f"数据集 {name} 不在支持列表中。支持的数据集: {list(DATASET_MAP.keys())}")
        
        # 更新参数
        args = ARG_MAP[name].copy()
        args.update(kwargs)
        
        # 获取缓存路径
        cache_dir = get_cache_path(DATASET_MAP[name])
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载数据集
        try:
            # 添加cache_dir参数以启用本地缓存
            dataset = load_dataset(
                DATASET_MAP[name],
                cache_dir=cache_dir,
                **args
            )
            
            if isinstance(dataset, dict):
                # 某些数据集可能返回DatasetDict，我们获取指定的split
                split = args.get('split', 'train')
                dataset = dataset[split]
            
            print(f"Dataset cache path: {cache_dir}")
            return dataset
            
        except Exception as e:
            raise Exception(f"加载数据集 {name} 时发生错误: {str(e)}")

def get_dataset(dt_name: str, **kwargs):
    """获取指定的数据集及其元信息

    Args:
        dt_name (str): 数据集名称
        **kwargs: 其他参数

    Returns:
        BaseDataSet: 包装后的数据集
    """
    dt = dataset_list(dt_name, **kwargs)
    meta = META_MAP.get(dt_name, {})
    return BaseDataSet(dt, meta)

if __name__ == "__main__":
    # 测试代码
    print("可用数据集列表:", dataset_list())
    
    # 测试加载IMDB数据集
    print("\n加载IMDB数据集...")
    imdb_dataset = get_dataset('imdb')
    print("IMDB数据集元信息:", imdb_dataset.meta)
    
    # 测试加载WikiText-2数据集
    print("\n加载WikiText-2数据集...")
    wiki_dataset = get_dataset('wikitext2')
    print("WikiText-2数据集元信息:", wiki_dataset.meta) 