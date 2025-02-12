import torchvision.models as models
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


CNN_MODELS_MAP = {
        'alexnet': models.alexnet,
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'densenet161': models.densenet161,
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19
        }


LM_MODELS_MAP = {
        'Llama-2-7b-hf': 'meta-llama/Llama-2-7b-hf',
        'Llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
        'DeepSeek-R1-Distill-Llama-8B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
}

MODELS_MAP = {**CNN_MODELS_MAP, **LM_MODELS_MAP}


def model_list(name:str=None):
    """Get model list or a specific model.

    Args:
        name (str, optional): Search a specific model or show all models(if None). Defaults to None.
    """
    if not name:
        return f"CNN based models: {list(CNN_MODELS_MAP.keys())}, LM models: {list(LM_MODELS_MAP.keys())}"
    else:
        try:
            return MODELS_MAP[name]
        except:
            raise ValueError(f"Make sure your name is in {list(MODELS_MAP.keys())}")

def get_model(model_name:str, pretrained:bool=True, num_classes:int=1000, *args, **kwargs):
    """Get a specific model.

    Args:
        model_name (str): Model name
        pretrained (bool, optional): If pretrained. Defaults to True.
        num_classes (int, optional): Number of classes. Defaults to 1000.
    """
    if model_name not in MODELS_MAP.keys() and not model_name.startswith('/'):
        raise ValueError(f"Make sure your name is in {list(MODELS_MAP.keys())} or a path to a model.")
    if model_name.startswith('/') or model_name in LM_MODELS_MAP.keys():
        try:
            model_name = LM_MODELS_MAP[model_name]
        except:
            pass
        model = AutoModelForCausalLM.from_pretrained(model_name, *args, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        setattr(model, '_model_type', 'LM')
        return model, tokenizer
    elif model_name in CNN_MODELS_MAP.keys():
        model = model_list(model_name)
        setattr(model, '_model_type', 'CNN')
        return model(pretrained=pretrained, num_classes=num_classes, *args, **kwargs)


if __name__ == "__main__":
    # print(model_list())
    import os
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    res = 'Llama-2-7b-hf'
    m = get_model(model_name=res, cache_dir='/mnt/share_data')
