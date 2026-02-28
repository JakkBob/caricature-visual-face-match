# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from PIL import Image
import logging
import clip
import json
import os
import gc
import random
import numpy as np

from custom_module import Baseline, BaselineCLIP, BaselineCLIPImageFusion, BaselineCLIPImageTextFusion
from loss import CrossModalInfoNCE
from dataset import SimpleUniqueIDLoader, GalleryDataset, ProbeDataset
from evaluate import validate_cross_modal


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 确保CUDNN的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_path):
    """配置日志系统，将日志同时输出到控制台和文件"""
    logger = logging.getLogger(f'Training_{os.path.basename(os.path.dirname(os.path.dirname(log_path)))}_{os.path.basename(os.path.dirname(log_path))}') # 使用唯一的名字
    logger.setLevel(logging.INFO)

    # 每次都清除已有的 handlers，确保日志文件独立
    if logger.handlers:
        logger.handlers.clear()

    # 创建一个 handler，用于写入日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 创建一个 handler, 用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义 handler 的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给 logger 添加 handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 在训练脚本的最开始调用
set_seed(2025)

dataset_names = ['WebCaricature', 'CaVI', 'IIIT-CFW']
# dataset_names = ['IIIT-CFW']

for dataset_name in dataset_names:
    # 数据集 || JSON 数据 || 日志
    dataset_dir = f'../../input/datasets/{dataset_name}'
    json_file = f'../../input/json/{dataset_name}/10-fold_cross-validation_split.json'
    kold_acc_dict = {}
    for kold in range(10):
        log_path = f'../../output4/{dataset_name}/kold-{str(kold+1)}/train.log'
        save_model_path = f'../../output4/{dataset_name}/kold-{str(kold+1)}/best_model.pth'

        output_dir = os.path.dirname(log_path)
        os.makedirs(output_dir, exist_ok=True)

        logger = setup_logger(log_path)

        logger.info(f"--- Dataset {dataset_name} kold={kold+1} Training Start ---")

        # 加载数据集划分
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)

        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像
            # transforms.RandomRotation(degrees=10),  # 随机旋转图像
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整图像的亮度和对比度 +++++
            # transforms.RandomGrayscale(p=0.1),  # 随机将彩色图像转换为灰度图像
            # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 随机应用透视变换，模拟从不同角度或位置观察对象的效果
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_clip = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        train_dataset = SimpleUniqueIDLoader(
            dataset_dir=dataset_dir,
            data_dict=loaded_json,
            kold=kold,
            batch_size=64,
            batch_num=20, # 您想要的批次数
            transform=transform,
            transform_clip=transform_clip,
            shuffle=True,
            seed=2025
        )
        gallery_dataset = GalleryDataset(dataset_dir=dataset_dir, data_dict=loaded_json, kold=kold, transform=transform, transform_clip=transform_clip)
        probe_dataset = ProbeDataset(dataset_dir=dataset_dir, data_dict=loaded_json, kold=kold, transform=transform, transform_clip=transform_clip)

        train_loader = DataLoader(
            train_dataset,
            batch_size=None,        # 【关键】因为我们的 iterable_dataset 已经在内部产生了批次
            num_workers=4, # 【关键】启用多进程
            pin_memory=True         # 【推荐】如果使用GPU，开启此选项可以加速数据从CPU到GPU的传输
        )
        gallery_loader = DataLoader(gallery_dataset, batch_size=128, shuffle=False, num_workers=4)
        probe_loader = DataLoader(probe_dataset, batch_size=128, shuffle=False, num_workers=4)

        # 模型
        # model = Baseline().to(device)
        # model = BaselineCLIP().to(device)
        # model = BaselineCLIPImageFusion().to(device)
        model = BaselineCLIPImageTextFusion().to(device)

        # 损失函数
        clip_loss_fn = CrossModalInfoNCE(temperature=0.07)
        classification_loss_fn = nn.CrossEntropyLoss()

        # 优化器与差异化学习率
        # 主干网络参数
        backbone_params = list(model.face_encoder.parameters()) + list(model.clip_model.visual.parameters())
        # 融合与投影头参数
        head_params = list(model.img_fusion_projector.parameters()) + list(model.text_fusion_projector.parameters())
        # params = list(model.classifier_head.parameters())
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-3},
        ], weight_decay=5e-4)

        backbone_milestones = [10, 20]
        head_milestones = [10, 20, 25]
        classifier_milestones = [5, 50]

        # scheduler = MultiStepLR(optimizer, milestones=[5, 10, 20, 25], gamma=0.1)

        epochs = 500
        best_c2p = 0.0
        # 早停的初始化参数
        patience = 10  # 容忍多少个epoch性能不提升
        acc_stop_count_patience = 5 # 容忍多少次准确率不变
        epochs_no_improve = 0  # 记录连续多少个epoch没有提升
        acc_stop_count = 0  # 记录多少次准确率没有变化
        for epoch in range(epochs):
            model.train()
            logger.info(f"------------------------- Epoch {epoch+1}/{epochs} -------------------------")
            running_clip_loss = 0.0
            train_loader.dataset.set_epoch(epoch)
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for caricature_img, caricature_img_clip, real_img, real_img_clip, identity, label in progress_bar:
                caricature_img = caricature_img.to(device)
                caricature_img_clip = caricature_img_clip.to(device)
                real_img = real_img.to(device)
                real_img_clip = real_img_clip.to(device)
                label = label.to(device)

                texts = [f"a photo of {i.lower().replace('_', ' ')}'s face" for i in identity]

                caricature_z, caricature_t = model(caricature_img, caricature_img_clip, texts)
                real_z, real_t = model(real_img, real_img_clip, texts)

                # Cross-Modal InfoNCE Loss
                # 分别计算漫画和真实人脸的对比损失
                loss_identity = clip_loss_fn(caricature_z, real_z)
                loss_clip_caricature = clip_loss_fn(caricature_z, real_t)
                loss_clip_real = clip_loss_fn(real_z, caricature_t)

                loss = 0.2 * loss_clip_caricature + 0.2 * loss_clip_real + 0.6 * loss_identity

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 转换为 FP32 进行优化
                for p in model.clip_model.parameters():
                    p.data = p.data.float()
                    if p.grad is not None:  # 关键修复：只处理有梯度的参数
                        p.grad.data = p.grad.data.float()
                
                optimizer.step()

                # 新增：转换权重回 FP16
                clip.model.convert_weights(model.clip_model)

                # 更新进度条
                progress_bar.set_postfix({
                    'Total': f'{loss.item():.4f}',
                    # 'Clip': f'{loss_clip.item():.4f}',
                    # 'Identity': f'{loss_identity.item():.4f}',
                    # 'Clasify': f'{loss_cls.item():.4f}',
                })

                running_clip_loss += loss.item()
            
            # scheduler.step()

            # 检查并更新 backbone 和 head 的学习率
            if epoch in backbone_milestones:
                optimizer.param_groups[0]['lr'] *= 0.1
                # print(f"  -> Milestone reached! New LR for {group.get('name', 'unnamed')}: {group['lr']:.6f}")
            
            # 检查并更新 backbone 和 head 的学习率
            if epoch in head_milestones:
                optimizer.param_groups[1]['lr'] *= 0.1
                # print(f"  -> Milestone reached! New LR for {group.get('name', 'unnamed')}: {group['lr']:.6f}")

            # # 检查并更新 classifier 的学习率
            # if epoch in classifier_milestones:
            #     optimizer.param_groups[2]['lr'] *= 0.1 # 第三个组是 classifier
            #     # print(f"  -> Milestone reached! New LR for Classifier: {optimizer.param_groups[2]['lr']:.6f}")
            

            # --- （可选）打印当前学习率以供调试 ---
            current_lr = optimizer.param_groups[0]['lr']
                
            logger.info(f"Epoch {epoch+1}/{epochs}, Clip Loss: {running_clip_loss/len(train_loader):.4f}, Current LR: {current_lr}")

            # 验证模型
            results = validate_cross_modal(model, probe_loader, gallery_loader, device, logger)

            if results['Rank-1'] > best_c2p:
                epochs_no_improve = 0
                best_c2p = results['Rank-1']
                # 保存最佳模型时
                # checkpoint = {
                #     'epoch': epoch + 1,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler_state_dict': scheduler.state_dict(),
                #     'best_c2p': best_c2p,
                # }
                # torch.save(model.state_dict(), save_model_path)
                logger.info(f"New best model saved with Rank-1: {best_c2p:.4f}, epoch: {epoch + 1}")
            elif results['Rank-1'] == best_c2p and acc_stop_count <= acc_stop_count_patience:
                acc_stop_count += 1
                epochs_no_improve = 0
                logger.info(f"New best model saved with Rank-1: {best_c2p:.4f}, epoch: {epoch + 1}")
            else:
                # 如果验证准确率没有提升
                epochs_no_improve += 1
                logger.info(f"早停机制记录-No improvement in validation accuracy. Patience: {epochs_no_improve}/{patience}")
            
            # 检查是否触发早停
            if epochs_no_improve >= patience:
                logger.info(f"早停触发-Early stopping triggered after {epoch + 1} epochs.")
                logger.info(f"Best Rank-1 accuracy achieved: {best_c2p:.4f}")
                kold_acc_dict[f'kold-{kold+1}'] = best_c2p
                break  # 跳出训练循环



        logger.info(f"--- Dataset {dataset_name} kold={kold+1} Training Finished ---")
        logger.info(f"--- Dataset {dataset_name} kold={kold+1} Best Rank-1: {best_c2p:.4f} ---")
        if kold + 1 == 10:
            average_accuracy = 0.0
            for key, value in kold_acc_dict.items():
                average_accuracy += value
                logger.info(f"--- Dataset {dataset_name} {key}: {value:.4f} ---")
            logger.info(f"--- Dataset {dataset_name} average_accuracy: {(average_accuracy/10):.4f} ---")

        # 1. 删除对模型和优化器的引用
        del model
        del optimizer
        # del scheduler
        
        # 2. 清空 PyTorch 的缓存，将显存真正归还给系统
        #    这在调试内存问题时非常有用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
