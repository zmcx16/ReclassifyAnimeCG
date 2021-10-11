import os
import glob
import random

from config import Config

if __name__ == '__main__':
    print('load config')
    cfg = Config()
    cfg.show()
    cfg_common = cfg.get()['config']['common']
    cfg_preprocess = cfg.get()['config']['preprocess']

    train_path = cfg_common['train_data_path']
    labels = os.listdir(train_path)
    label_path = cfg_common['label_path']

    train_ratio = cfg_preprocess['train_ratio']

    index_label_path = os.path.join(label_path, 'index.txt')
    if os.path.isfile(index_label_path):
        os.remove(index_label_path)

    train_label_path = os.path.join(label_path, 'train.txt')
    if os.path.isfile(train_label_path):
        os.remove(train_label_path)

    test_label_path = os.path.join(label_path, 'test.txt')
    if os.path.isfile(test_label_path):
        os.remove(test_label_path)

    total_images = 0
    print('\ngenerate data index files:')
    for index, label in enumerate(labels):
        with open(index_label_path, 'a') as f:
            f.write(str(index) + '|' + label)
            f.write('\n')

        img_list = []
        for file_type in ('*.png', '*.jpg', '*.bmp'):
            img_list.extend(glob.glob(os.path.join(train_path, label, file_type)))

        total_images += len(img_list)

        random.shuffle(img_list)

        train_list = img_list[:int(train_ratio * len(img_list))]
        test_list = img_list[len(train_list):]

        print('%d|%s: tr(%d), ts(%d)' % (index, label, len(train_list), len(test_list)))

        with open(train_label_path, 'a') as f:
            for img in train_list:
                f.write(img + '|' + str(index))
                f.write('\n')

        with open(test_label_path, 'a') as f:
            for img in test_list:
                f.write(img + '|' + str(index))
                f.write('\n')

    print('total images: ', total_images)
    print('\npreprocess done')
