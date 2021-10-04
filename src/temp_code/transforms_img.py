import torchvision.transforms as trns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image_path = 'I:\\work\\WORK\\ReclassifyAnimeCG\\ReclassifyAnimeCG\\data-sample\\classified-data\\Rem\\2.jpg'
"""
transforms = trns.Compose([
    trns.Resize((224, 224)),
    trns.ToTensor(),
    trns.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])
"""
"""
transforms = trns.Compose([
    trns.Resize((224, 224)),
    trns.ToTensor()])
"""
transforms = trns.Compose([
    trns.Resize((224, 224)),
    trns.ToTensor()
])

# Read image and run prepro
image = Image.open(image_path).convert("RGB")
image_tensor = transforms(image)
print(type(image_tensor), image_tensor.shape)

# So we need to reshape it to (H, W, C):
image_tensor = image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[0])
print(type(image_tensor), image_tensor.shape)

plt.imshow(image_tensor)
plt.show()

# image_tensor = image_tensor.unsqueeze(0)
# print(image_tensor.size())
