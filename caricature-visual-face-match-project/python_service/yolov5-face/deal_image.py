from tqdm import tqdm
import cv2
import numpy as np
import copy
import os


# image_paths = './output_image_add.txt'
# cari_img = []
# with open(image_paths, 'r') as file:
#     lines = file.readlines()
#     cari_img = [line.strip() for line in lines]

# for path in cari_img:
#     # 读取图像
#     image = cv2.imread(path)
#     # 将BGR图像转换为RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = copy.deepcopy(image)
#     # 指定目标大小
#     target_size = (112, 112)
#     # 调整裁剪后的图像大小到指定大小
#     resized_image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
#     # save_path = 'D:/DSW_code/face_detect/datasets/output'
#     # img_name = path.split('\\')[-1]
#     # # 保存图像
#     # cv2.imwrite(os.path.join(save_path, img_name), cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
#     cv2.imwrite(path, cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

data_dir = './runs/detect/exp6/WebCaricature'
output_dir = 'D:/DSW_code/face_detect/datasets/output/WebCaricature'
for identity in tqdm(os.listdir(data_dir)):
    for img_name in os.listdir(os.path.join(data_dir, identity)):
        if 'C' in img_name:
            save_path = os.path.join(output_dir, 'Caricature', identity)
        if 'P' in img_name:
            save_path = os.path.join(output_dir, 'Real', identity)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 读取图像
        image = cv2.imread(os.path.join(data_dir, identity, img_name))
        # 将BGR图像转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = copy.deepcopy(image)
        # 保存图像
        cv2.imwrite(os.path.join(save_path, img_name), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))