# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from facenet_pytorch import InceptionResnetV1

class ModelNet(nn.Module):
    def __init__(self, device='cuda', embedding_dim=512, num_classes=None):
        super(SAFENet, self).__init__()
        self.device = device
        
        self.face_encoder = InceptionResnetV1(pretrained="casia-webface", classify=False, device=device) # casia-webface vggface2
        face_feat_dim = self.face_encoder.last_linear.out_features # 512 for InceptionResnetV1
        
        self.freeze_backbone()

        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip_feat_dim = self.clip_model.visual.output_dim # 512 for ViT-B/32

        # 确保模型以 FP16 初始化
        clip.model.convert_weights(self.clip_model)  # 初始化后立即转换

        self.img_fusion_projector = nn.Sequential(
            nn.Linear(face_feat_dim + clip_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )

        self.text_fusion_projector = nn.Sequential(
            nn.Linear(face_feat_dim + clip_feat_dim + clip_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )
        
        self.text_encoder = self.clip_model.encode_text

        self.classifier_head = nn.Linear(embedding_dim, num_classes)
    
    def freeze_backbone(self):
        """冻结 facenet 的前面层"""
        for name, param in self.face_encoder.named_parameters():
            if 'repeat_3' not in name and 'block8' not in name and 'last_linear' not in name and 'last_bn' not in name:
                param.requires_grad = False

    def forward(self, images, img_clip, texts):
        """
        前向传播
        Args:
            images (torch.Tensor): 输入图像批次
            img_clip (torch.Tensor): 输入图像批次 (CLIP ViT)
            texts (list): 对应的文本标签列表
        Returns:
            torch.Tensor: 融合后的图像特征向量 z
            torch.Tensor: 融合后的带文本语义的向量 t
            torch.Tensor: 分类头输出 logits
        """

        f_face = self.face_encoder(images)
        
        f_clip = self.clip_model.visual(img_clip.half()).float()
        
        # 图像特征融合
        f_concat = torch.cat((f_face, f_clip), dim=1)
        z = self.img_fusion_projector(f_concat)
        
        # 文本编码
        text_tokens = clip.tokenize(texts).to(self.device)
        t = self.text_encoder(text_tokens).float()

        # 图像-文本特征融合
        t_concat = torch.cat((f_concat, t), dim=1)
        t = self.text_fusion_projector(t_concat)
        t = torch.nn.functional.normalize(t, dim=-1)

        logits = self.classifier_head(z)
        
        # 对特征进行L2归一化，便于计算余弦相似度
        z = torch.nn.functional.normalize(z, dim=-1)

        return z, t, logits

    def get_embeddings(self, images, img_clip, texts=None):
        """ 仅用于推理，获取图像的融合特征z """
        with torch.no_grad():
            f_face = self.face_encoder(images)
            f_clip = self.clip_model.visual(img_clip.half()).float()
            f_concat = torch.cat((f_face, f_clip), dim=1)
            z = self.img_fusion_projector(f_concat)
            z = torch.nn.functional.normalize(z, dim=-1)
        
        if texts is not None:
            # 文本编码
            text_tokens = clip.tokenize(texts).to(self.device)
            t = self.text_encoder(text_tokens).float()

            # 图像-文本特征融合
            t_concat = torch.cat((f_concat, t), dim=1)
            t = self.text_fusion_projector(t_concat)
            t = torch.nn.functional.normalize(t, dim=-1)
            return t

        return z
