import os
import glob
import random

from config import Config

if __name__ == '__main__':
    print('load config')
    cfg = Config()
    cfg.show()
    cfg_obj = cfg.get()['config']

    train_path = cfg_obj['train_data_path']
    labels = os.listdir(train_path)

    cv_train_ratio = cfg_obj['cv_train_ratio']
    cv_val_ratio = cfg_obj['cv_val_ratio']

    label_path = cfg_obj['label_path']

    index_label_path = os.path.join(label_path, 'index.txt')
    if os.path.isfile(index_label_path):
        os.remove(index_label_path)

    train_label_path = os.path.join(label_path, 'train.txt')
    if os.path.isfile(train_label_path):
        os.remove(train_label_path)

    val_label_path = os.path.join(label_path, 'val.txt')
    if os.path.isfile(val_label_path):
        os.remove(val_label_path)

    test_label_path = os.path.join(label_path, 'test.txt')
    if os.path.isfile(test_label_path):
        os.remove(test_label_path)

    for index, label in enumerate(labels):
        with open(index_label_path, 'a') as f:
            f.write(str(index) + '|' + label)
            f.write('\n')

        img_list = []
        for file_type in ('*.png', '*.jpg', '*.bmp'):
            img_list.extend(glob.glob(os.path.join(train_path, label, file_type)))

        random.shuffle(img_list)
        print(len(img_list))
        train_list = img_list[:int(cv_train_ratio*len(img_list))]
        val_list = img_list[len(train_list):int((cv_train_ratio+cv_val_ratio)*len(img_list))]
        test_list = img_list[len(train_list) + len(val_list):]
        with open(train_label_path, 'a') as f:
            for img in train_list:
                f.write(img + '|' + str(index))
                f.write('\n')

        with open(val_label_path, 'a') as f:
            for img in val_list:
                f.write(img + '|' + str(index))
                f.write('\n')

        with open(test_label_path, 'a') as f:
            for img in test_list:
                f.write(img + '|' + str(index))
                f.write('\n')
