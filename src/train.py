import sys
import os
import torch.nn as nn
import torch.optim as optim
import time

from data import *
from config import Config
from utils import adjust_learning_rate_step
from models.default_model import DefaultModel


def save_model(use_gpu_num, model, save_path):
    if use_gpu_num > 1:
        checkpoint = {'model': model.module,
                      'model_state_dict': model.module.state_dict(),
                      # 'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, save_path)
    else:
        checkpoint = {'model': model,
                      'model_state_dict': model.state_dict(),
                      # 'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, save_path)


if __name__ == "__main__":
    start_time = time.perf_counter()

    print('load config')
    cfg = Config()
    cfg.show()
    cfg_common = cfg.get()['config']['common']
    cfg_train = cfg.get()['config']['train']
    cfg_models_parameter = cfg.get()['config']['models_parameter']

    label_path = cfg_common['label_path']
    model_name = cfg_common['use_model_name']
    save_folder = os.path.join(cfg_common['train_model_path'], model_name)
    use_gpu_num = cfg_common['use_gpu_num']

    start_save_best_model = cfg_train['start_save_best_model']

    model_cfg = cfg_models_parameter[model_name]
    optimizer_algo = model_cfg['optimizer_algo']
    loss_algo = model_cfg['loss_algo']
    batch_size = model_cfg['batch_size']
    image_input_size = model_cfg['image_input_size']
    max_epoch = model_cfg['max_epoch']
    resume_epoch = model_cfg['resume_epoch']
    weight_decay = model_cfg['weight_decay']
    momentum = model_cfg['momentum']
    lr = model_cfg['lr']

    model = DefaultModel().load_model(cfg.get()['config'])
    # print(model)

    if use_gpu_num > 1:
        print('****** using multiple gpus to training ********')
        model = nn.DataParallel(model, device_ids=list(range(use_gpu_num)))
    else:
        print('****** using single gpu to training ********')
    print("...... Initialize the network done!!! .......")

    if torch.cuda.is_available():
        model.cuda()

    if optimizer_algo == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        # optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_algo == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        # criterion = LabelSmoothSoftmaxCE()
        # criterion = LabelSmoothingCrossEntropy()

    train_datasets, train_dataloader = get_datasets_and_dataloader(os.path.join(label_path, 'train.txt'), 'train', batch_size, image_input_size)

    epoch_size = len(train_datasets) // batch_size

    max_iter = max_epoch * epoch_size
    start_iter = resume_epoch * epoch_size
    epoch = resume_epoch

    warmup_epoch = 5
    warmup_steps = warmup_epoch * epoch_size
    global_step = 0

    stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
    step_index = 0

    min_loss = sys.float_info.max

    model.train()
    for iteration in range(start_iter, max_iter):
        global_step += 1

        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(train_dataloader)
            loss = 0
            epoch += 1
            if epoch % 5 == 0 and epoch > 0:
                save_model(use_gpu_num, model, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate_step(optimizer, model_cfg['lr'], 0.1, epoch, step_index, iteration, epoch_size)

        # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
        #                           learning_rate_base=cfg.LR,
        #                           total_steps=max_iter,
        #                           warmup_steps=warmup_steps)

        # try:
        images, labels = next(batch_iterator)
        # except:
        #     continue

        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        out = model(images)

        labels = labels.to(dtype=torch.int64)
        # labels = torch.tensor(labels, dtype=torch.long)

        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = torch.max(out, 1)[1]
        train_correct = (prediction == labels).sum()

        # print(train_correct.type())
        train_acc = (train_correct.float()) / batch_size
        loss_val = loss.item()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % loss_val + 'ACC: %.3f ||' % (train_acc * 100) + 'LR: %.8f' % (lr))

        if epoch >= max_epoch * start_save_best_model and loss_val < min_loss:
            min_loss = loss_val
            print('got min loss {}, update the final.pth'.format(min_loss))
            save_model(use_gpu_num, model, os.path.join(save_folder, 'final.pth'))

    seconds = time.time() - start_time
    print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))
