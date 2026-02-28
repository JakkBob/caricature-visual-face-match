from tqdm import tqdm
import cv2
import numpy as np
import copy
import os

# 裁剪图像并保存
def crop_save(modal_img, identity, points_dir, data_dir, output_dir):
    for img_name in modal_img:
        assert ('P' in img_name) or ('C' in img_name), f"文件名格式不规范: {img_name}"
        points_file = img_name.split('.')[0] + '.txt'
        # 读取关键点坐标
        x_min, y_min, x_max, y_max = 0, 0, 0, 0
        with open(os.path.join(points_dir, identity, points_file), 'r') as file:
            lines = file.readlines()
            points = [[float(line.strip().split(' ')[0]), float(line.strip().split(' ')[1])] for line in lines ]
            points = np.array(points)
            x1, y1, x2, y2 = points[1][0], points[0][1], points[3][0], points[2][1]
            # 计算裁剪区域的边界
            x_min = int(np.floor(x1.min()))
            y_min = int(np.floor(y1.min()))
            x_max = int(np.ceil(x2.max()))
            y_max = int(np.ceil(y2.max()))

        # 读取图像
        image = cv2.imread(os.path.join(data_dir, identity, img_name))
        # 将BGR图像转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 确保边界在图像尺寸范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        img = copy.deepcopy(image)
        # 根据关键点坐标裁剪图像
        cropped_image = img[y_min:y_max, x_min:x_max]
        # 指定目标大小
        target_size = (112, 112)
        # 调整裁剪后的图像大小到指定大小
        resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)

        if 'C' in img_name:
            save_path = os.path.join(output_dir, 'Caricature', identity)
        if 'P' in img_name:
            save_path = os.path.join(output_dir, 'Real', identity)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存图像
        cv2.imwrite(os.path.join(save_path, img_name), cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

data_dir = 'D:/DSW_code/face_detect/datasets/WebCaricature/OriginalImages'
points_dir = 'D:/DSW_code/face_detect/datasets/WebCaricature/FacialPoints'
file_path = 'D:/DSW_code/face_detect/datasets/WebCaricature/Filenames'
output_dir = 'D:/DSW_code/face_detect/datasets/output/WebCaricature'
for identity in tqdm(os.listdir(file_path)):
    cari_paths = os.path.join(file_path, identity, 'file_c.txt')
    real_paths = os.path.join(file_path, identity, 'file_p.txt')
    # 处理漫画图像
    cari_img = []
    with open(cari_paths, 'r') as file:
        lines = file.readlines()
        cari_img = [line.strip() for line in lines]
    crop_save(cari_img, identity, points_dir, data_dir, output_dir)
    # 处理人脸图像
    real_img = []
    with open(real_paths, 'r') as file:
        lines = file.readlines()
        real_img = [line.strip() for line in lines]
    crop_save(real_img, identity, points_dir, data_dir, output_dir)