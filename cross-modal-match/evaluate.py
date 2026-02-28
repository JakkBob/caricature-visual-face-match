import torch
from tqdm import tqdm
import numpy as np

def validate_cross_modal(model, probe_loader, gallery_loader, device, logger):
    """
    为每个跨模态测试场景计算并报告独立的 Rank-n 准确率。
    注意：此函数不会计算场景平均值，因为各场景评估协议不同。
    
    Args:
        model: 训练好的特征提取模型。
        probe_loader: 包含Probe数据的DataLoader。
        gallery_loader: 包含Gallery数据的DataLoader。
        device: 计算设备。

    Returns:
        一个字典，键为场景描述，值为该场景的准确率字典。
        例如: {"漫画->照片": {"Rank-1": 0.85, ...}, "照片->漫画": {...}}
    """
    logger.info(f"------------------------ 开始跨模态验证 ------------------------")
    model.eval()
    
    # 用于存储每个场景的结果，以便返回
    results_summary = {}

    with torch.no_grad():    
        # --- (数据加载和预处理部分保持不变) ---
        raw_probe_features, raw_probe_ids = [], []
        raw_gallery_features, raw_gallery_ids = [], []
        for image, image_clip, identity, label in tqdm(probe_loader, desc=f"处理 Probe 数据", leave=False):
            image = image.to(device)
            image_clip = image_clip.to(device)
            # features = model(image)
            # features = model(image, image_clip)
            # features = model(image, image_clip)
            features = model.get_embeddings(image, image_clip)
            raw_probe_features.append(features)
            raw_probe_ids.extend(identity)

        for image, image_clip, identity, label in tqdm(gallery_loader, desc=f"处理 Gallery 数据", leave=False):
            image = image.to(device)
            image_clip = image_clip.to(device)
            texts = [f"a photo of {i.lower().replace('_', ' ')}'s face" for i in identity]
            # features = model(image)
            # features = model(image, image_clip)
            # features = model(image, image_clip)
            features = model.get_embeddings(image, image_clip, texts)
            raw_gallery_features.append(features)
            raw_gallery_ids.extend(identity)
            
        probe_features = torch.cat(raw_probe_features, dim=0)
        gallery_features = torch.cat(raw_gallery_features, dim=0)

        all_string_ids = set(raw_probe_ids + raw_gallery_ids)
        id_to_label = {id_str: i for i, id_str in enumerate(sorted(list(all_string_ids)))}
        logger.info(f"发现 {len(id_to_label)} 个唯一身份。")
        logger.info(f"Probe 数据包含 {len(raw_probe_ids)} 个样本。")
        logger.info(f"Gallery 数据包含 {len(raw_gallery_ids)} 个样本。")

        # --- 检查ID不匹配情况 ---
        probe_only_ids = set(raw_probe_ids) - set(raw_gallery_ids)
        if probe_only_ids:
            logger.warning(f"警告: 有 {len(probe_only_ids)} 个 Probe ID 在 Gallery 中不存在。")


        probe_labels_np = np.array([id_to_label[id_str] for id_str in raw_probe_ids], dtype=np.int64)
        gallery_labels_np = np.array([id_to_label[id_str] for id_str in raw_gallery_ids], dtype=np.int64)
        probe_labels = torch.from_numpy(probe_labels_np).to(device)
        gallery_labels = torch.from_numpy(gallery_labels_np).to(device)

        # --- 使用 cumsum 方法灵活计算 Rank-n 准确率 ---
        
        # 1. 定义你想要计算的 Rank 列表
        ranks_to_compute = [1, 5, 10]  # 你可以在这里修改，例如 [1, 3, 5, 10, 20]
        
        # 2. 计算相似度矩阵和排序索引
        similarity_matrix = torch.mm(probe_features, gallery_features.t())
        _, sort_indices = torch.sort(similarity_matrix, dim=1, descending=True)
        
        # 3. 获取排序后的匹配情况（布尔矩阵）
        correct_matches = (probe_labels[:, None] == gallery_labels[sort_indices])
        
        # 4. 使用累积求和计算在每个位置之前，总共找到了多少个正确匹配
        cumulative_correct = correct_matches.float().cumsum(dim=1)
        
        # 5. 计算每个指定 Rank 的准确率
        scenario_results = {}
        for rank in ranks_to_compute:
            # 获取当前 Rank 对应的索引（例如 Rank-1 -> 索引 0）
            # 使用 min() 防止索引越界，非常安全
            idx = min(rank - 1, cumulative_correct.size(1) - 1)
            
            # 如果累积和大于0，说明在当前Rank及之前找到了匹配
            accuracy = (cumulative_correct[:, idx] > 0).float().mean().item()
            scenario_results[f'Rank-{rank}'] = accuracy

        # --- 打印并存储当前协议的结果 ---
        logger.info(f"------------------------ 跨模态验证结果 ------------------------")
        for rank_name, acc in scenario_results.items():
            logger.info(f"{rank_name:<10} {acc:.4f}")
        logger.info("---------------------------------------------------------------")
        
        # 将结果存入字典
        results_summary = scenario_results

    return results_summary
