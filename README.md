# ReclassifyAnimeCG

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e1c5c6bc5a7e4299b6e36db92aea9206)](https://app.codacy.com/gh/zmcx16/ReclassifyAnimeCG?utm_source=github.com&utm_medium=referral&utm_content=zmcx16/ReclassifyAnimeCG&utm_campaign=Badge_Grade_Settings)

Using PyTorch to reclassify & organize your anime cg, the PyTorch skeleton code of this repository is base on [lxztju / pytorch_classification](https://github.com/lxztju/pytorch_classification) and make the user can setup environment and use PyTorch vision defult model easily.
Besides, This tool support the useful function to make the user can easy organize the classified image files.

# Requirements
1. Install the PyTorch(base on your computer & GPU environment): https://pytorch.org/
2. Install the packages in the requirements.txt

# Configuration

```
config:
  common:
    use_model_name: "resnext50_32x4d" # choose the model which define on models_parameter
    use_gpu_num: 1
    train_data_path: "{absolute path of the trainig data}" # (e.g. D:/ReclassifyAnimeCG/data-sample/train)
    train_model_path: "{absolute path of the training model store path}" # (e.g. D:/ReclassifyAnimeCG/data-sample/train_weight)
    label_path: "{absolute path of the preprocess data info store path}" # (e.g. D:/ReclassifyAnimeCG/data-sample/data-sample)
    output_data_path: "{absolute path of the output images store path}" # (e.g. D:/ReclassifyAnimeCG/data-sample/output)
    predict_data_path: "I:/work/WORK/ReclassifyAnimeCG/ReclassifyAnimeCG/data-sample/predict" # use when predict_use_test_txt = false, predict your specific images instead of testing data from training data (e.g. D:/ReclassifyAnimeCG/data-sample/predict)
  preprocess:
    train_ratio: 0.7 # generate train_ratio training data from train_data_path, and generate (1-train_ratio) testing data from train_data_path
    save_training_data_info_in_txt: true # save the training data information when preprocess (e.g. the crc32 info for training data images) 
    delete_duplicate_images_on_training_data: true # use when save_training_data_info_in_txt = true, the preprocess script will delete duplicate images on training data (delete on disks)
  train:
    start_save_best_model: 0.7 # try to save best model when current epoch >= start_save_best_model * max_epoch
  predict:
    use_test_txt: true # set false if you want to predict your specific images instead of testing data from training data
    ignore_predict_file_in_training_data_info: true # use when use_test_txt=false & train_info.json exist (save_training_data_info_in_txt=true)
    copy_predict_result_to_output_path: true # copy the predict image to output folder by predict result label
    copy_predict_result_use_symbolic_link: true # use when copy_predict_result_to_output_path=true, create the symbolic link instead of really image file
  models_parameter:
    {use_model_name}: # all support model reference on models_parameter of config.yaml
      use_default_pretrained_model: true # download and use the pretrained model from web
      use_final_model: false # training model use the previous training best model
      optimizer_algo: 'SGD'
      loss_algo: 'CrossEntropyLoss'
      batch_size: 32
      image_input_size: 300
      max_epoch: 25
      resume_epoch: 0
      weight_decay: 5.e-4
      momentum: 0.9
      lr: 1.e-3
      freeze_first_n_children: 0
      freeze_first_n_parameters: 0
```
# Support Model
| Model | Support List |
| ------ | ------ |
| AlexNet |[alexnet]|
| Densenet |[densenet121, densenet169, densenet201, densenet161]|
| EfficientNet |[efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7]|
| MNASNet |[mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3]|
| MobileNet |[mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small]|
| regnet |[regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, regnet_y_8gf, regnet_y_16gf, regnet_y_32gf, regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf, regnet_x_8gf, regnet_x_16gf, regnet_x_32gf]|
| ResNet / WideResNet / ResNeXt |[resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2]|
| ShuffleNet |[shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0]|
| SqueezeNet |[squeezenet1_0, squeezenet1_1]|
| VGG |[vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn]|
| lbpcascade_animeface + CNN |TBD|

# How to use
1. Setup the config.yaml for your requirement
2. Run the preprocess.py (ReclassifyAnimeCG/src/preprocess/preprocess.py), the preprocess.py will generate train.txt, test.txt, index.txt on the label_path
3. Run the train.py (ReclassifyAnimeCG/src/train.py), the train.py will train the model and store on train_model_path
4. Run the predict.py (ReclassifyAnimeCG/src/predict.py), the predict.py will predict the images / copy the image to output base on the config.yaml setting.
5. (Optional) If you want to organize your images rather than turn the classifier model, check the output images, delete the wrong predict result on the output, and then copy the output to your training data folder
6. (Optional) Jump to Step2, keep going until you organize your images completed.

Note. If you set copy_predict_result_use_symbolic_link=true, the Windows OS system can't preview images on File explorer, if you want to organize predict result easily, I recommand the [FastStone Image Viewer](https://www.faststone.org/FSViewerDetail.htm) tool, it can view the image file easily even the image file is symbolic link.

# Evaluation 
Evaluation model performance using the ReclassifyAnimeCG data-sample (tr: 225 ts: 100) and non-tuning parameter
| Model | Acc |
| ------ | ------ |
| alexnet |77|
| densenet169 |92|
| efficientnet-b3 |91|
| mnasnet1_0 |64|
| mobilenet_v2 |87|
| regnet_x_16gf |93|
| resnext50_32x4d |92|
| shufflenet_v2_x2_0 |44|
| squeezenet1_1 |87|

For the PyTorch default model evaluation, reference: 
https://pytorch.org/vision/stable/models.html#classification

# Reference
1. lxztju / pytorch_classification - (https://github.com/lxztju/pytorch_classification)
2. nagadomi / lbpcascade_animeface - (https://github.com/nagadomi/lbpcascade_animeface)


# License
This project is licensed under the terms of the MIT license.
