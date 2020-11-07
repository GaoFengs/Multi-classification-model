import glob

import numpy as np
import cv2
import os
import  torchvision
from PIL import Image
from scipy import misc
from scipy import ndimage

#“”“该程序作用是将单通道图像转成3通道”""
# i_dir = 'D:\pyWork\PET\data\train\PET40000'
#
# for name in os.listdir(i_dir):
#     print(name)
#     image_path = os.path.join(i_dir,name)
#     img = cv2.imread(image_path,0)
#     image = np.expand_dims(img, axis=2)
#     image = np.concatenate((image, image, image), axis=-1)
#     base_name = name.split(',')[0]
#     #print(base_name)
#     file = os.path.join(i_dir, name)  # 必须拼接完整文件名
#     os.remove(file)
#     print(file + " remove succeeded")
#     cv2.imwrite(i_dir + '\\' + base_name,image)

# data_path = ['D:\pyWork\PET\data\\train\AD','D:\pyWork\PET\data\\train\CN','D:\pyWork\PET\data\\test\AD&CN']
# for path in data_path:
#     img_names = os.listdir(path)
#     for image_name in img_names:
#         image = Image.open(path + '\\' + image_name)
#         image_resize = torchvision.transforms.Resize([168,168], interpolation=2)
#         image = image_resize(image)
#         print(path + "_168")
#         image.save(path + "_168" + '\\' + image_name)

##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    print(filenumber)
    rate = 0.3  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    print(picknumber)
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(len(set(sample)))
    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)
    return


if __name__ == '__main__':
    for i in range(5):
        os.mkdir('D:\pyWork\PET\data\\train\\train\\result' + str(i))
        fileDir = "D:\pyWork\PET\data\\train\\train\AD\\"  # 源图片文件夹路径
        tarDir = 'D:\pyWork\PET\data\\train\\train\\result' + str(i) + '\\'  # 移动到新的文件夹路径
        moveFile(fileDir)
        fileDir = "D:\pyWork\PET\data\\train\\train\CN\\"  # 源图片文件夹路径
        tarDir = 'D:\pyWork\PET\data\\train\\train\\result' + str(i) + '\\'  # 移动到新的文件夹路径
        moveFile(fileDir)
















