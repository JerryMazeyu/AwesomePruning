import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")

MODELS_MAP = {
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


def model_list(name:str=None):
    """Get model list or a specific model.

    Args:
        name (str, optional): Search a specific model or show all models(if None). Defaults to None.
    """
    if not name:
        return list(MODELS_MAP.keys())
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
    model = model_list(model_name)
    return model(pretrained=pretrained, num_classes=num_classes, *args, **kwargs)

if __name__ == "__main__":
    print(model_list())
    res = 'resnet18'
    m = get_model(model_name=res)
    print(m)
