import os
import torch.nn as nn
import torch
import requests
from collections import OrderedDict
from models import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl \
    , resnet50, resnet101, densenet121, densenet169, mobilenet_v2, EfficientNet


ModelFuncDict = {
    'resnext101_32x8d': resnext101_32x8d_wsl,
    'resnext101_32x16d': resnext101_32x16d_wsl,
    'resnext101_32x32d': resnext101_32x32d_wsl,
    'resnext101_32x48d': resnext101_32x48d_wsl,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'moblienetv2': mobilenet_v2,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'efficientnet-b0': EfficientNet,
    'efficientnet-b1': EfficientNet,
    'efficientnet-b2': EfficientNet,
    'efficientnet-b3': EfficientNet,
    'efficientnet-b4': EfficientNet,
    'efficientnet-b5': EfficientNet,
    'efficientnet-b6': EfficientNet,
    'efficientnet-b7': EfficientNet
}


def download(url, path):
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)


def build_default_model(num_classes, model_name, model_path, model_url, test=False):
    if model_name.startswith('efficientnet'):
        return build_efficientnet_model(num_classes, model_name, model_path, model_url, test=False)
    elif model_name.startswith('resn'):
        return build_resnet_resnext_model(num_classes, model_name, model_path, model_url, test=False)
    elif model_name.startswith('densenet'):
        return build_densenet_model(num_classes, model_name, model_path, model_url, test=False)

    print('Not mapping expect model')
    return None


def build_resnet_resnext_model(num_classes, model_name, model_path, model_url, test=False):
    model = ModelFuncDict[model_name]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def build_efficientnet_model(num_classes, model_name, model_path, model_url, test=False):
    model = ModelFuncDict[model_name].from_name(model_name)

    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    fc_features = model.get_fc().in_features
    model.set_fc(nn.Linear(fc_features, num_classes))
    return model


def build_densenet_model(num_classes, model_name, model_path, model_url, test=False):
    model = ModelFuncDict[model_name]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # fix keys
            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +\
                    '.'+k.split('.')[-1]
            # print(k)
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model
