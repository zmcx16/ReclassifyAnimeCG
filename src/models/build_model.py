import os
import torch.nn as nn
import torch
import requests
from collections import OrderedDict
from models import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl \
    , resnet50, resnet101, densenet121, densenet169, mobilenet_v2, alexnet, googlenet, EfficientNet


ModelFuncDict = {
    'resnext101_32x8d': resnext101_32x8d_wsl,
    'resnext101_32x16d': resnext101_32x16d_wsl,
    'resnext101_32x32d': resnext101_32x32d_wsl,
    'resnext101_32x48d': resnext101_32x48d_wsl,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'alexnet': alexnet,
    'moblienetv2': mobilenet_v2,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet121': densenet121,
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

    # wrapper the build model to always use pretraining model
    # (don't want to modify pytorch model code and don't want download pretraining model everytime)
    build_model_search_pattern_mapping_func_dict = {
        'resn': build_resnet_resnext_model,
        'alexnet': build_alexnet_model,
        'moblienet': build_moblienet_model,
        'densenet': build_densenet_model,
        'efficientnet': build_efficientnet_model
    }

    for search_key in build_model_search_pattern_mapping_func_dict:
        if model_name.startswith(search_key):
            return build_model_search_pattern_mapping_func_dict[search_key](num_classes, model_name, model_path, model_url, test=False)

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


def build_moblienet_model(num_classes, model_name, model_path, model_url, test=False):
    model = ModelFuncDict[model_name]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    return model


def build_alexnet_model(num_classes, model_name, model_path, model_url, test=False):
    model = ModelFuncDict[model_name]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )

    return model
