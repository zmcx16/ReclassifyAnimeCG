import os
import abc
import torch

from models import resnext101_32x8d
ModelFuncDict = {
    'resnext101_32x8d': resnext101_32x8d
}


def load_checkpoint(file_path):
    check_point = torch.load(file_path)
    m = check_point['model']
    m.load_state_dict(check_point['model_state_dict'])
    return m


class DefaultModel(abc.ABC):
    model_name = ""
    model_cfg = {}
    resume_epoch = 0

    def load_model(self, cfg):
        self.model_name = cfg['use_model_name']
        self.model_cfg = cfg['models_parameter'][self.model_name]
        self.resume_epoch = self.model_cfg['resume_epoch']
        model_url = self.model_cfg['model_url']

        save_folder = os.path.join(cfg['train_model_path'], cfg['use_model_name'])
        os.makedirs(save_folder, exist_ok=True)
        train_model_path = os.path.join(save_folder, self.model_name+'.pth')

        # build the network model
        if not self.model_cfg['resume_epoch']:
            print('****** Training {} ****** '.format(self.model_name))
            print('****** loading the Imagenet pretrained weights ****** ')
            model = ModelFuncDict[self.model_name](num_classes=4, model_path=train_model_path, model_url=model_url)
            ct = 0
            for child in model.children():
                ct += 1
                # print(child)
                if ct < 8:
                    print(child)
                    for param in child.parameters():
                        param.requires_grad = False
            # print(model)
        if self.resume_epoch:
            print(' ******* Resume training from {}  epoch {} *********'.format(self.model_name, self.resume_epoch))
            model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(self.resume_epoch)))
        return model
