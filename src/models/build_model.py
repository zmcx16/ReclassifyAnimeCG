import os
import torch.nn as nn
import torch
import requests
from models import resnext101_32x8d_wsl


def download(url, path):
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)


def resnext101_32x8d(num_classes, model_path, model_url, test=False):
    model = resnext101_32x8d_wsl()
    if not test:
        if not os.path.isfile(model_path):
            print('download model from ', model_url)
            download(model_url, model_path)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model
