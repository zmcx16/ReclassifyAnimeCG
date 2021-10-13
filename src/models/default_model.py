import os
import abc
import torch
from models import build_default_model


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
        cfg_common = cfg['common']
        cfg_models_parameter = cfg['models_parameter']

        self.model_name = cfg_common['use_model_name']
        self.model_cfg = cfg_models_parameter[self.model_name]
        self.resume_epoch = self.model_cfg['resume_epoch']

        save_folder = os.path.join(cfg_common['train_model_path'], cfg_common['use_model_name'])
        os.makedirs(save_folder, exist_ok=True)
        #train_model_path = os.path.join(save_folder, self.model_name+'.pth')

        index_label_path = os.path.join(cfg_common['label_path'], 'index.txt')
        with open(index_label_path,  'r')as f:
            num_classes = len(f.readlines())

        # build the network model
        if not self.model_cfg['resume_epoch']:
            print('****** Training {} ****** '.format(self.model_name))
            print('****** loading the Imagenet pretrained weights ****** ')
            model = build_default_model(model_name=self.model_name, num_classes=num_classes)
            # print(model)
            print('children:')
            freeze_first_n_children = self.model_cfg['freeze_first_n_children']
            freeze_first_n_parameters = self.model_cfg['freeze_first_n_parameters']
            c = 0
            ct = 0
            for name_m, child in model.named_children():
                ct += 1
                print('child.named: ', name_m)
                for name_p, param in child.named_parameters():
                    c += 1
                    print('parameter.named: ', name_p)
                    if ct < freeze_first_n_children or c < freeze_first_n_parameters:
                        param.requires_grad = False

            print('total module children: ', ct)
            print('total parameters: ', c)
            # print(model)
        if self.resume_epoch:
            print(' ******* Resume training from {}  epoch {} *********'.format(self.model_name, self.resume_epoch))
            model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(self.resume_epoch)))
        return model
