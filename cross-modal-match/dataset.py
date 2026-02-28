import json
import random
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from PIL import Image
import os
from typing import List, Dict, Tuple, Callable, Optional

from torchvision import transforms

class SimpleUniqueIDLoader(IterableDataset):
    """
    一个极简的、可迭代的数据加载器。
    它同时扮演了Dataset和Sampler的角色，确保每个批次内的身份是唯一的。
    使用方式与标准的DataLoader几乎完全一样。
    """
    def __init__(self,
                 dataset_dir: str,
                 data_dict: Dict,
                 kold: int,
                 batch_size: int,
                 batch_num: int,
                 transform: Optional[Callable] = None,
                 transform_clip: Optional[Callable] = None,
                 shuffle: bool = True,
                 seed: int = 2025):
        
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.shuffle = shuffle
        self.transform = transform
        self.transform_clip = transform_clip
        self.seed = seed
        self.epoch = 0 # 内部epoch计数器
        
        # 从data_dict中提取所需信息
        self.identities = data_dict['splits'][kold]['train']['identities']
        self.train_samples = data_dict['splits'][kold]['train']['samples']

        assert self.batch_size <= len(self.identities), "批次大小不能大于总身份数"
        
        # 创建身份到标签的映射
        self.identity_to_label = {identity: idx for idx, identity in enumerate(self.identities)}

    def set_epoch(self, epoch):
        """在每个epoch开始前调用此方法，以更新随机种子"""
        self.epoch = epoch

    def _load_and_transform(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载单张图片并应用两种变换"""
        img = Image.open(os.path.join(self.dataset_dir, path)).convert('RGB')

        if self.transform:
            img_tensor = self.transform(img)

        if self.transform_clip:
            img_clip_tensor = self.transform_clip(img)

        return img_tensor, img_clip_tensor
    
    def __iter__(self):
        """
        核心逻辑：根据指定的 batch_num 生成批次的迭代器。
        每次生成批次前，都会重新随机打乱所有身份。
        """
        # 为多进程环境设置不同的种子
        worker_info = get_worker_info()
        if worker_info is None:  # 单进程模式
            batch_num = self.batch_num
            worker_id = 0
        else:  # 多进程模式
            # 计算每个worker应该生成的批次数
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            # 确保总批次数能被worker数整除，或者让最后一个worker处理剩余部分
            # 这里我们使用整除，让每个worker处理相同的数量
            batch_num = self.batch_num // total_workers

        # 结合基础种子、epoch和worker_id生成唯一种子
        seed = self.seed + self.epoch * 1000 + worker_id
        # 设置随机种子以保证可复现性
        rnd = random.Random(seed)

        num_identities = len(self.identities)
        
        # 循环 batch_num 次，生成指定数量的批次
        for _ in range(batch_num):
            # 每次都重新创建并打乱身份索引列表
            identity_indices = list(range(num_identities))
            if self.shuffle:
                rnd.shuffle(identity_indices)
            
            # 选取前 batch_size 个身份作为当前批次的身份
            batch_identity_indices = identity_indices[:self.batch_size]

            # 准备一个批次的数据
            batch_c_imgs, batch_c_imgs_clip, batch_r_imgs, batch_r_imgs_clip = [], [], [], []
            batch_identities, batch_labels = [], []
            
            for id_idx in batch_identity_indices: 
                identity = self.identities[id_idx]
                sample_paths = self.train_samples[identity]
                
                c_path = rnd.choice(sample_paths['caricature'])
                r_path = rnd.choice(sample_paths['real'])
                
                c_img, c_img_clip = self._load_and_transform(c_path)
                r_img, r_img_clip = self._load_and_transform(r_path)
                
                batch_c_imgs.append(c_img)
                batch_c_imgs_clip.append(c_img_clip)
                batch_r_imgs.append(r_img)
                batch_r_imgs_clip.append(r_img_clip)
                
                batch_identities.append(identity)
                batch_labels.append(self.identity_to_label[identity])

            yield (
                torch.stack(batch_c_imgs),
                torch.stack(batch_c_imgs_clip),
                torch.stack(batch_r_imgs),
                torch.stack(batch_r_imgs_clip),
                batch_identities,
                torch.tensor(batch_labels)
            )

    def __len__(self):
        """返回一个epoch中的批次数"""
        return self.batch_num


class ProbeDataset(Dataset):
    """探针数据集加载器 - 包含所有漫画图像和身份文本"""
    def __init__(self, dataset_dir: str, data_dict: Dict, kold: int, transform=None, transform_clip=None):
        self.dataset_dir = dataset_dir
        self.data_dict = data_dict
        self.transform = transform
        self.transform_clip = transform_clip
        
        # 获取测试集的身份和样本
        self.identities = data_dict['splits'][kold]['test']['identities']
        self.samples = data_dict['splits'][kold]['test']['samples']
        
        # 创建身份到标签的映射
        self.identity_to_label = {identity: idx for idx, identity in enumerate(self.identities)}
        
        # 预处理数据：收集所有漫画图像
        self.data_list = []
        for identity in self.identities:
            sample = self.samples[identity]
            for caricature_path in sample['caricature']:
                self.data_list.append({
                    'identity': identity,
                    'caricature_path': os.path.join(self.dataset_dir, caricature_path),
                    'label': self.identity_to_label[identity]
                })
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        # 加载图像
        img = Image.open(data['caricature_path']).convert('RGB')

        if self.transform:
            image = self.transform(img)
        
        if self.transform_clip:
            image_clip = self.transform_clip(img)
        
        return image, image_clip,  data['identity'], data['label']


class GalleryDataset(Dataset):
    """画廊数据集加载器 - 每个身份只选取一张图像"""
    def __init__(self, dataset_dir: str, data_dict: Dict, kold: int, transform=None, transform_clip=None, seed=2025):
        self.dataset_dir = dataset_dir
        self.data_dict = data_dict
        self.transform = transform
        self.transform_clip = transform_clip
        self.seed = seed
        
        # 获取测试集的身份和样本
        self.identities = data_dict['splits'][kold]['test']['identities']
        self.samples = data_dict['splits'][kold]['test']['samples']
        
        # 创建身份到标签的映射
        self.identity_to_label = {identity: idx for idx, identity in enumerate(self.identities)}
        
        # 预处理数据：每个身份随机选择一张图像
        rnd = random.Random(seed) # 创建一个独立的随机数生成器
        self.data_list = []
        for identity in self.identities:
            sample = self.samples[identity]
            real_path = rnd.choice(sample['real'])
            
            self.data_list.append({
                'identity': identity,
                'real_path': os.path.join(self.dataset_dir, real_path),
                'label': self.identity_to_label[identity]
            })
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        # 加载图像
        img = Image.open(data['real_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(img)
        
        if self.transform_clip:
            image_clip = self.transform_clip(img)
        
        return image, image_clip, data['identity'], data['label']
