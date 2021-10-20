import os
import re
import torch.nn as nn
import torch
import requests
from collections import OrderedDict

from .torchvision_models.alexnet import alexnet, model_urls as alexnet_model_urls
from .torchvision_models.densenet import densenet121, densenet169, densenet201, densenet161, model_urls as densenet_model_urls
from .efficientnet_pytorch.model import EfficientNet, model_urls as efficientnet_model_urls
from .torchvision_models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3, _MODEL_URLS as mnasnet_model_urls
from .torchvision_models.mobilenet import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, model_urls as mobilenet_model_urls
from .torchvision_models.regnet import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, \
    regnet_y_8gf, regnet_y_16gf, regnet_y_32gf, regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf, \
    regnet_x_8gf, regnet_x_16gf, regnet_x_32gf, model_urls as regnet_model_urls
from .torchvision_models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, \
    resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, model_urls as resnet_model_urls
from .torchvision_models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, \
    shufflenet_v2_x2_0, model_urls as shufflenet_model_urls
from .torchvision_models.squeezenet import squeezenet1_0, squeezenet1_1, model_urls as squeezenet_model_urls
from .torchvision_models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, \
    model_urls as vgg_model_urls


def download(url, path):
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)


# wrapper the build model to always use pretraining model
# (don't want to modify pytorch model code and don't want download pretraining model everytime)
def build_default_model(num_classes, model_name, model_path, pretrained=True, test=False):
    build_model_dict = {
        'alexnet': {"model_param": {"model": alexnet, "url": alexnet_model_urls["alexnet"]}, "build_func": build_alexnet_model},
        "densenet121": {"model_param": {"model": densenet121, "url": densenet_model_urls["densenet121"]}, "build_func": build_densenet_model},
        "densenet169": {"model_param": {"model": densenet169, "url": densenet_model_urls["densenet169"]}, "build_func": build_densenet_model},
        "densenet201": {"model_param": {"model": densenet201, "url": densenet_model_urls["densenet201"]}, "build_func": build_densenet_model},
        "densenet161": {"model_param": {"model": densenet161, "url": densenet_model_urls["densenet161"]}, "build_func": build_densenet_model},
        "efficientnet-b0": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b0"]}, "build_func": build_efficientnet_model},
        "efficientnet-b1": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b1"]}, "build_func": build_efficientnet_model},
        "efficientnet-b2": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b2"]}, "build_func": build_efficientnet_model},
        "efficientnet-b3": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b3"]}, "build_func": build_efficientnet_model},
        "efficientnet-b4": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b4"]}, "build_func": build_efficientnet_model},
        "efficientnet-b5": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b5"]}, "build_func": build_efficientnet_model},
        "efficientnet-b6": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b6"]}, "build_func": build_efficientnet_model},
        "efficientnet-b7": {"model_param": {"model": EfficientNet, "name": model_name, "url": efficientnet_model_urls["efficientnet-b7"]}, "build_func": build_efficientnet_model},
        "mnasnet0_5": {"model_param": {"model": mnasnet0_5, "url": mnasnet_model_urls["mnasnet0_5"]}, "build_func": build_mnasnet_model},
        "mnasnet0_75": {"model_param": {"model": mnasnet0_75, "url": mnasnet_model_urls["mnasnet0_75"]}, "build_func": build_mnasnet_model},
        "mnasnet1_0": {"model_param": {"model": mnasnet1_0, "url": mnasnet_model_urls["mnasnet1_0"]}, "build_func": build_mnasnet_model},
        "mnasnet1_3": {"model_param": {"model": mnasnet1_3, "url": mnasnet_model_urls["mnasnet1_3"]}, "build_func": build_mnasnet_model},
        "mobilenet_v2": {"model_param": {"model": mobilenet_v2, "url": mobilenet_model_urls["mobilenet_v2"]}, "build_func": build_moblienet_model},
        "mobilenet_v3_large": {"model_param": {"model": mobilenet_v3_large, "url": mobilenet_model_urls["mobilenet_v3_large"]}, "build_func": build_moblienet_model},
        "mobilenet_v3_small": {"model_param": {"model": mobilenet_v3_small, "url": mobilenet_model_urls["mobilenet_v3_small"]}, "build_func": build_moblienet_model},
        "regnet_y_400mf": {"model_param": {"model": regnet_y_400mf, "url": regnet_model_urls["regnet_y_400mf"]}, "build_func": build_regnet_model},
        "regnet_y_800mf": {"model_param": {"model": regnet_y_800mf, "url": regnet_model_urls["regnet_y_800mf"]}, "build_func": build_regnet_model},
        "regnet_y_1_6gf": {"model_param": {"model": regnet_y_1_6gf, "url": regnet_model_urls["regnet_y_1_6gf"]}, "build_func": build_regnet_model},
        "regnet_y_3_2gf": {"model_param": {"model": regnet_y_3_2gf, "url": regnet_model_urls["regnet_y_3_2gf"]}, "build_func": build_regnet_model},
        "regnet_y_8gf": {"model_param": {"model": regnet_y_8gf, "url": regnet_model_urls["regnet_y_8gf"]}, "build_func": build_regnet_model},
        "regnet_y_16gf": {"model_param": {"model": regnet_y_16gf, "url": regnet_model_urls["regnet_y_16gf"]}, "build_func": build_regnet_model},
        "regnet_y_32gf": {"model_param": {"model": regnet_y_32gf, "url": regnet_model_urls["regnet_y_32gf"]}, "build_func": build_regnet_model},
        "regnet_x_400mf": {"model_param": {"model": regnet_x_400mf, "url": regnet_model_urls["regnet_x_400mf"]}, "build_func": build_regnet_model},
        "regnet_x_800mf": {"model_param": {"model": regnet_x_800mf, "url": regnet_model_urls["regnet_x_800mf"]}, "build_func": build_regnet_model},
        "regnet_x_1_6gf": {"model_param": {"model": regnet_x_1_6gf, "url": regnet_model_urls["regnet_x_1_6gf"]}, "build_func": build_regnet_model},
        "regnet_x_3_2gf": {"model_param": {"model": regnet_x_3_2gf, "url": regnet_model_urls["regnet_x_3_2gf"]}, "build_func": build_regnet_model},
        "regnet_x_8gf": {"model_param": {"model": regnet_x_8gf, "url": regnet_model_urls["regnet_x_8gf"]}, "build_func": build_regnet_model},
        "regnet_x_16gf": {"model_param": {"model": regnet_x_16gf, "url": regnet_model_urls["regnet_x_16gf"]}, "build_func": build_regnet_model},
        "regnet_x_32gf": {"model_param": {"model": regnet_x_32gf, "url": regnet_model_urls["regnet_x_32gf"]}, "build_func": build_regnet_model},
        "resnet18": {"model_param": {"model": resnet18, "url": resnet_model_urls["resnet18"]}, "build_func": build_resnet_resnext_model},
        "resnet34": {"model_param": {"model": resnet34, "url": resnet_model_urls["resnet34"]}, "build_func": build_resnet_resnext_model},
        "resnet50": {"model_param": {"model": resnet50, "url": resnet_model_urls["resnet50"]}, "build_func": build_resnet_resnext_model},
        "resnet101": {"model_param": {"model": resnet101, "url": resnet_model_urls["resnet101"]}, "build_func": build_resnet_resnext_model},
        "resnet152": {"model_param": {"model": resnet152, "url": resnet_model_urls["resnet152"]}, "build_func": build_resnet_resnext_model},
        "resnext50_32x4d": {"model_param": {"model": resnext50_32x4d, "url": resnet_model_urls["resnext50_32x4d"]}, "build_func": build_resnet_resnext_model},
        "resnext101_32x8d": {"model_param": {"model": resnext101_32x8d, "url": resnet_model_urls["resnext101_32x8d"]}, "build_func": build_resnet_resnext_model},
        "wide_resnet50_2": {"model_param": {"model": wide_resnet50_2, "url": resnet_model_urls["wide_resnet50_2"]}, "build_func": build_resnet_resnext_model},
        "wide_resnet101_2": {"model_param": {"model": wide_resnet101_2, "url": resnet_model_urls["wide_resnet101_2"]}, "build_func": build_resnet_resnext_model},
        "shufflenet_v2_x0_5": {"model_param": {"model": shufflenet_v2_x0_5, "url": shufflenet_model_urls["shufflenetv2_x0.5"]}, "build_func": build_shufflenet_model},
        "shufflenet_v2_x1_0": {"model_param": {"model": shufflenet_v2_x1_0, "url": shufflenet_model_urls["shufflenetv2_x1.0"]}, "build_func": build_shufflenet_model},
        "shufflenet_v2_x1_5": {"model_param": {"model": shufflenet_v2_x1_5, "url": shufflenet_model_urls["shufflenetv2_x1.5"]}, "build_func": build_shufflenet_model},
        "shufflenet_v2_x2_0": {"model_param": {"model": shufflenet_v2_x2_0, "url": shufflenet_model_urls["shufflenetv2_x2.0"]}, "build_func": build_shufflenet_model},
        "squeezenet1_0": {"model_param": {"model": squeezenet1_0, "url": squeezenet_model_urls["squeezenet1_0"]}, "build_func": build_squeezenet_model},
        "squeezenet1_1": {"model_param": {"model": squeezenet1_1, "url": squeezenet_model_urls["squeezenet1_1"]}, "build_func": build_squeezenet_model},
        "vgg11":  {"model_param": {"model": vgg11, "url": vgg_model_urls["vgg11"]}, "build_func": build_vgg_model},
        "vgg13":  {"model_param": {"model": vgg13, "url": vgg_model_urls["vgg13"]}, "build_func": build_vgg_model},
        "vgg16":  {"model_param": {"model": vgg16, "url": vgg_model_urls["vgg16"]}, "build_func": build_vgg_model},
        "vgg19":  {"model_param": {"model": vgg19, "url": vgg_model_urls["vgg19"]}, "build_func": build_vgg_model},
        "vgg11_bn":  {"model_param": {"model": vgg11_bn, "url": vgg_model_urls["vgg11_bn"]}, "build_func": build_vgg_model},
        "vgg13_bn":  {"model_param": {"model": vgg13_bn, "url": vgg_model_urls["vgg13_bn"]}, "build_func": build_vgg_model},
        "vgg16_bn":  {"model_param": {"model": vgg16_bn, "url": vgg_model_urls["vgg16_bn"]}, "build_func": build_vgg_model},
        "vgg19_bn":  {"model_param": {"model": vgg19_bn, "url": vgg_model_urls["vgg19_bn"]}, "build_func": build_vgg_model},
    }
    if model_name in build_model_dict:
        return build_model_dict[model_name]["build_func"](num_classes, model_path,
                                                              build_model_dict[model_name]["model_param"], pretrained, test)

    print('Not mapping expect model')
    return None


def build_alexnet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
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


def build_densenet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
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


def build_efficientnet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    model_name = model_param["name"]
    model = model_param["model"].from_name(model_name)
    if not test or not pretrained or not model_url:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    fc_features = model.get_fc().in_features
    model.set_fc(nn.Linear(fc_features, num_classes))
    return model


def build_mnasnet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, num_classes))
    return model


def build_moblienet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.set_classifier(num_classes)
    return model


def build_regnet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.fc = nn.Linear(in_features=model.current_width, out_features=num_classes)
    return model


def build_resnet_resnext_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def build_shufflenet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.fc = nn.Linear(model._stage_out_channels[-1], num_classes)
    return model


def build_squeezenet_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5), nn.Conv2d(512, model.num_classes, kernel_size=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
    )
    return model


def build_vgg_model(num_classes, model_path, model_param, pretrained, test):
    model_url = model_param["url"]
    if not pretrained or not model_url:
        return model_param["model"](num_classes=num_classes)

    model = model_param["model"]()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes),
    )
    return model
