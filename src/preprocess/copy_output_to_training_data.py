import os
import glob
import shutil

from config import Config


if __name__ == '__main__':
    print('load config')
    cfg = Config()
    cfg.show()
    cfg_common = cfg.get()['config']['common']

    train_path = cfg_common['train_data_path']
    output_data_path = cfg_common['output_data_path']
    labels = os.listdir(output_data_path)

    for index, label in enumerate(labels):
        img_list = []
        for file_type in ('*.png', '*.jpg', '*.bmp'):
            img_list.extend(glob.glob(os.path.join(output_data_path, label, file_type)))

        for _, img_path in enumerate(img_list):
            file_name, extension = os.path.basename(img_path).rsplit('.', 1)
            duplicate_i = 0
            duplicate_text = ''
            while True:
                final_path = os.path.join(train_path, label, file_name + duplicate_text + '.' + extension)
                if not os.path.isfile(final_path):
                    break
                duplicate_i += 1
                duplicate_text = ' (' + str(duplicate_i) + ')'

            print('copy {} to {}'.format(img_path, final_path))
            if os.path.islink(img_path):
                linkto = os.readlink(img_path)
                os.symlink(linkto, final_path)
            else:
                shutil.copy(img_path, final_path)
