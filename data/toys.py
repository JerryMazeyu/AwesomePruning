from base import BaseDataSet
import torchvision.datasets as datasets
from config import CONF
import warnings
warnings.filterwarnings("ignore")

DATASET_MAP = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
        'fashionmnist':datasets.FashionMNIST
        }

ARG_MAP = {
        'cifar10': {'root': CONF.data_root_path, 'train': True, 'download': True},
        'cifar100': {'root': CONF.data_root_path, 'train': True, 'download': True},
        'mnist': {'root': CONF.data_root_path, 'train': True, 'download': True},
        'fashionmnist': {'root': CONF.data_root_path, 'train': True, 'download': True},
}

META_MAP = {
        'cifar10': {'img_size': 'example', 'description': "cifar10 dataset."},
}


def dataset_list(name:str=None, **kwargs):
    """Get dataset list or a specific dataset.

    Args:
        name (str, optional): Search a specific dataset or show all dataset(if None). Defaults to None.
    """
    if not name:
        return list(DATASET_MAP.keys())
    else:
        ARG_MAP[name].update(kwargs)
        return DATASET_MAP[name](**ARG_MAP[name])

def get_dataset(dt_name:str, **kwargs):
    """Get a specific model.

    Args:
        dt_name (str): Dataset name.
    """
    dt = dataset_list(dt_name, **kwargs)
    meta = META_MAP.get(dt_name, {})
    return BaseDataSet(dt, meta)


if __name__ == "__main__":
    print(dataset_list())
    data_name = "cifar10"
    dt = dataset_list(data_name, train=False)
    dt_with_meta = get_dataset(data_name, train=False)
    print(dt_with_meta.meta)

