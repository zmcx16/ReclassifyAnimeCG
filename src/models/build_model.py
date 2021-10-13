import os
import re
import torch.nn as nn
import torch
import requests

from .torchvision_models.alexnet import alexnet, model_urls as alexnet_model_urls
from .torchvision_models.densenet import densenet121, densenet169, densenet201, densenet161
from .torchvision_models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, model_urls as efficientnet_model_urls
from .torchvision_models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from .torchvision_models.mobilenet import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
from .torchvision_models.regnet import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, \
    regnet_y_8gf, regnet_y_16gf, regnet_y_32gf, regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf, \
    regnet_x_8gf, regnet_x_16gf, regnet_x_32gf
from .torchvision_models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, \
    resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from .torchvision_models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, \
    shufflenet_v2_x2_0
from .torchvision_models.squeezenet import squeezenet1_0, squeezenet1_1
from .torchvision_models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


def download(url, path):
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)


# wrapper the build model to always use pretraining model
# (don't want to modify pytorch model code and don't want download pretraining model everytime)
def build_default_model(num_classes, model_name, model_path, pretrained=True, test=False):
    build_model_dict = {
        'alexnet': {"model": alexnet, "url": alexnet_model_urls["alexnet"], "reset_num_classes": reset_alexnet_num_classes},
        "densenet121": densenet121,
        "densenet169": densenet169,
        "densenet201": densenet201,
        "densenet161": densenet161,
        "efficientnet_b0": efficientnet_b0,
        "efficientnet_b1": efficientnet_b1,
        "efficientnet_b2": efficientnet_b2,
        "efficientnet_b3": {"model": efficientnet_b3, "url": efficientnet_model_urls["efficientnet_b3"], "reset_num_classes": None},
        "efficientnet_b4": efficientnet_b4,
        "efficientnet_b5": efficientnet_b5,
        "efficientnet_b6": efficientnet_b6,
        "efficientnet_b7": efficientnet_b7,
        "mnasnet0_5": mnasnet0_5,
        "mnasnet0_75": mnasnet0_75,
        "mnasnet1_0": mnasnet1_0,
        "mnasnet1_3": mnasnet1_3,
        "mobilenet_v2": mobilenet_v2,
        "mobilenet_v3_large": mobilenet_v3_large,
        "mobilenet_v3_small": mobilenet_v3_small,
        "regnet_y_400mf": regnet_y_400mf,
        "regnet_y_800mf": regnet_y_800mf,
        "regnet_y_1_6gf": regnet_y_1_6gf,
        "regnet_y_3_2gf": regnet_y_3_2gf,
        "regnet_y_8gf": regnet_y_8gf,
        "regnet_y_16gf": regnet_y_16gf,
        "regnet_y_32gf": regnet_y_32gf,
        "regnet_x_400mf": regnet_x_400mf,
        "regnet_x_800mf": regnet_x_800mf,
        "regnet_x_1_6gf": regnet_x_1_6gf,
        "regnet_x_3_2gf": regnet_x_3_2gf,
        "regnet_x_8gf": regnet_x_8gf,
        "regnet_x_16gf": regnet_x_16gf,
        "regnet_x_32gf": regnet_x_32gf,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "resnext50_32x4d": resnext50_32x4d,
        "resnext101_32x8d": resnext101_32x8d,
        "wide_resnet50_2": wide_resnet50_2,
        "wide_resnet101_2": wide_resnet101_2,
        "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
        "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
        "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
        "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
        "squeezenet1_0": squeezenet1_0,
        "squeezenet1_1": squeezenet1_1,
        "vgg11": vgg11,
        "vgg13": vgg13,
        "vgg16": vgg16,
        "vgg19": vgg19,
        "vgg11_bn": vgg11_bn,
        "vgg13_bn": vgg13_bn,
        "vgg16_bn": vgg16_bn,
        "vgg19_bn": vgg19_bn,
    }
    if model_name in build_model_dict:
        model_dict = build_model_dict[model_name]
        if pretrained:
            model = model_dict["model"]()
            if not test:
                if not os.path.isfile(model_path):
                    print('download model from ', model_dict["url"])
                    download(model_dict["url"], model_path)

                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)

            model_dict["reset_num_classes"](num_classes, model)
        else:
            model = model_dict["model"](num_classes)

        return model

    print('Not mapping expect model')
    return None


def reset_alexnet_num_classes(num_classes, model):
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
