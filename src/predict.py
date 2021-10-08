import os
import glob
import torch
import shutil
from PIL import Image
import pandas as pd
from tqdm import tqdm
from config import Config
from data import get_test_transform


def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict(model, imgs, image_input_size):

    model = load_checkpoint(model)
    print('..... Finished loading model! ......')

    if torch.cuda.is_available():
        model.cuda()
    img_path_list, predict_list, actual_list = [], [], []
    correct_num = 0
    for i in tqdm(range(len(imgs))):
        img_path, actual = imgs[i].split('|')
        actual = int(actual.strip())
        actual_list.append(actual)
        img_path = img_path.strip()
        # print(img_path)
        img_path_list.append(img_path)
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(size=image_input_size)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        predict_list.append(prediction)

        if prediction == actual:
            correct_num += 1

    return img_path_list, predict_list, actual_list, correct_num


if __name__ == "__main__":

    print('load config')
    cfg = Config()
    cfg.show()
    cfg_obj = cfg.get()['config']
    label_path = cfg_obj['label_path']
    index_label_path = os.path.join(label_path, 'index.txt')
    copy_predit_result_to_output_path = cfg_obj['copy_predit_result_to_output_path']
    output_data_path = cfg_obj['output_data_path']

    model_name = cfg_obj['use_model_name']
    model_cfg = cfg_obj['models_parameter'][model_name]
    model_folder = os.path.join(cfg_obj['train_model_path'], model_name)
    trained_model_path = os.path.join(model_folder, 'final.pth')
    image_input_size = model_cfg['image_input_size']

    predict_use_test_txt = cfg_obj['predict_use_test_txt']
    if predict_use_test_txt:
        test_label_path = os.path.join(label_path, 'test.txt')
        with open(test_label_path,  'r')as f:
            imgs = f.readlines()
    else:
        # generate imgs list
        predict_data_path = cfg_obj['predict_data_path']
        img_list = []
        for file_type in ('*.png', '*.jpg', '*.bmp'):
            img_list.extend(glob.iglob(os.path.join(predict_data_path, file_type)))

        imgs = []
        for img in img_list:
            imgs.append(img + '|' + str(-1))

    img_path_list, predict_list, actual_list, correct_num = predict(trained_model_path, imgs, image_input_size)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    submission = pd.DataFrame({"img_path_list": img_path_list, "predict": predict_list, "actual": actual_list})
    print(submission)
    #submission.to_csv(cfg.BASE + '{}_submission.csv'.format(model_name), index=False, header=False)

    print()
    if predict_use_test_txt:
        print('ACC: %.2f' % (correct_num * 100.0 / len(imgs))+'%')

    if copy_predit_result_to_output_path:
        print('copy all predict images to output')
        label_dict = {}
        with open(index_label_path,  'r')as f:
            for line in f:
                label_index, label_text = line.split('|')
                label_dict[label_index] = label_text.strip()

        for index, img_path in enumerate(img_path_list):

            file_name, extension = os.path.basename(img_path).split('.')
            duplicate_i = 0
            duplicate_text = ''
            while True:
                output_path = os.path.join(output_data_path, label_dict[str(predict_list[index])], file_name + duplicate_text + '.' + extension)
                if not os.path.isfile(output_path):
                    break
                duplicate_i += 1
                duplicate_text = ' (' + duplicate_i + ')'

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(img_path, output_path)
