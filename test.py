import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import h5py as h5
from torchvision.utils import save_image

# def find_char(char, char_list):
#     for i in range(len(char_list)):
#         if char == char_list[i]:
#             return True
#             break
#         else:
#             pass
#         return False
#
#
# char_path = r"C:\Users\苏永亮\Desktop\font.txt"
#
# with open(r"C:\Users\苏永亮\Desktop\font.txt", 'r') as f:
#     chara = f.read()
#
#
# char_list = list(chara)
#
#
# img_path = 'FZ-GroundTruth100-128/书体坊柳公权楷简190'
# img_names = [i for i in os.listdir(img_path)]
#
# char_del_list = []
# for i in range(len(img_names)):
#     char = bytes((r'\u' + img_names[i].split('.')[0][4:8].lower()).encode()).decode("unicode_escape")
#     a = find_char(char, char_list)
#     if char in char_list:
#         pass
#
#     else:
#         char_del_list.append(char)
#
# print(char_del_list)
#
#
# root_path = './FZ-GroundTruth100-128'
# font_list = [i for i in os.listdir(root_path)]
# for font_name in font_list:
#     for i in range(len(char_del_list)):
#         char_unicode = char_del_list[i].encode('unicode-escape').decode()
#         char_unicode = char_unicode[2:6].upper()
#         img_name = 'U_00' + char_unicode + '.jpg'
#         img_del_path = os.path.join(root_path, font_name, img_name)
#         os.remove(img_del_path)
# img_path = r'D:\Datasets\Font-470\source\U_004E8B.jpg'
# image = Image.open(img_path)
# tranform = transforms.Compose([
#
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.FiveCrop(128)
# ])
# image = tranform(image)
# print(image[0].shape)


# img_path = r'D:\Datasets\Font-470\source\U_004E8B.jpg'
# image = Image.open(img_path)
# trf = transforms.ToTensor()
# image = trf(image)
#
# mask = torch.ones((1, 3, 256, 256))
# rdn_1 = np.random.randint(1, 150)
# rdn_2 = np.random.randint(1, 150)
# mask[:, :, rdn_1:(rdn_1 + 100), rdn_2:(rdn_2 + 100)] = 0
#
# image_mask = torch.mul(image, mask)
# save_image(image_mask, 'mask.png')
queue = torch.zeros((8, 0, 512))
a = torch.randn(8, 1, 512)
b = torch.cat((queue, a), dim=1)
print(b.shape)
