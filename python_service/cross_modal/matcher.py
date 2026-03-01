"""
Cross-Modal Face Matching Model
Based on cross-modal-match from the repository
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union
from PIL import Image

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[Warning] CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("[Warning] facenet-pytorch not available. Install with: pip install facenet-pytorch")


class CrossModalFaceMatcher(nn.Module):
    """
    Cross-Modal Face Matching Model.
    
    This model combines face features from FaceNet and visual features from CLIP
    to perform cross-modal matching between caricatures and real face photos.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        embedding_dim: int = 512,
        num_classes: Optional[int] = None,
        pretrained_face: str = 'casia-webface'
    ):
        """
        Initialize Cross-Modal Face Matcher.
        
        Args:
            device: Device to run model ('cuda' or 'cpu')
            embedding_dim: Dimension of the final embedding
            num_classes: Number of classes for classification (optional)
            pretrained_face: Pretrained weights for FaceNet ('casia-webface' or 'vggface2')
        """
        super(CrossModalFaceMatcher, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Face encoder (InceptionResnetV1 from facenet-pytorch)
        if FACENET_AVAILABLE:
            self.face_encoder = InceptionResnetV1(
                pretrained=pretrained_face,
                classify=False,
                device=self.device
            )
            face_feat_dim = 512  # InceptionResnetV1 output dimension
            self._freeze_face_encoder()
        else:
            self.face_encoder = None
            face_feat_dim = 512
            print("[Warning] Face encoder not available. Using placeholder.")
        
        # CLIP model
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32",
                device=self.device,
                jit=False
            )
            clip_feat_dim = self.clip_model.visual.output_dim  # 512 for ViT-B/32
            
            # Convert CLIP weights to FP16 only on CUDA
            if self.device.type == 'cuda':
                clip.model.convert_weights(self.clip_model)
            else:
                # On CPU, keep FP32
                self.clip_model = self.clip_model.float()
        else:
            self.clip_model = None
            self.clip_preprocess = None
            clip_feat_dim = 512
            print("[Warning] CLIP model not available. Using placeholder.")
        
        # Image fusion projector (FaceNet + CLIP visual features)
        self.img_fusion_projector = nn.Sequential(
            nn.Linear(face_feat_dim + clip_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )
        
        # Text fusion projector (for gallery images with text)
        self.text_fusion_projector = nn.Sequential(
            nn.Linear(face_feat_dim + clip_feat_dim + clip_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )
        
        # Text encoder (use CLIP's text encoder)
        if self.clip_model is not None:
            self.text_encoder = self.clip_model.encode_text
        else:
            self.text_encoder = None
        
        # Classification head (optional)
        if num_classes is not None:
            self.classifier_head = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier_head = None
        
        self.embedding_dim = embedding_dim
    
    def _freeze_face_encoder(self):
        """Freeze early layers of FaceNet, keep later layers trainable."""
        for name, param in self.face_encoder.named_parameters():
            if 'repeat_3' not in name and 'block8' not in name and 'last_linear' not in name and 'last_bn' not in name:
                param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        img_clip: torch.Tensor,
        texts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            images: Face images for FaceNet (B, 3, 160, 160)
            img_clip: Images for CLIP (B, 3, 224, 224)
            texts: Optional text descriptions for gallery images
        
        Returns:
            z: Fused image features (B, embedding_dim)
            t: Fused text features (B, embedding_dim) if texts provided
            logits: Classification logits if classifier head exists
        """
        # Face features
        if self.face_encoder is not None:
            f_face = self.face_encoder(images)
        else:
            f_face = torch.zeros(images.size(0), 512, device=self.device)
        
        # CLIP visual features
        if self.clip_model is not None:
            # Use half precision only on CUDA
            if self.device.type == 'cuda':
                f_clip = self.clip_model.visual(img_clip.half()).float()
            else:
                f_clip = self.clip_model.visual(img_clip.float()).float()
        else:
            f_clip = torch.zeros(img_clip.size(0), 512, device=self.device)
        
        # Image feature fusion
        f_concat = torch.cat((f_face, f_clip), dim=1)
        z = self.img_fusion_projector(f_concat)
        
        # Text encoding (optional)
        t = None
        if texts is not None and self.text_encoder is not None:
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            t = self.text_encoder(text_tokens).float()
            
            # Image-text feature fusion
            t_concat = torch.cat((f_concat, t), dim=1)
            t = self.text_fusion_projector(t_concat)
            t = F.normalize(t, dim=-1)
        
        # Classification (optional)
        logits = None
        if self.classifier_head is not None:
            logits = self.classifier_head(z)
        
        # L2 normalize features
        z = F.normalize(z, dim=-1)
        
        return z, t, logits
    
    def get_embeddings(
        self,
        images: torch.Tensor,
        img_clip: torch.Tensor,
        texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Get embeddings for inference.
        
        Args:
            images: Face images for FaceNet (B, 3, 160, 160)
            img_clip: Images for CLIP (B, 3, 224, 224)
            texts: Optional text descriptions for gallery images
        
        Returns:
            Embeddings (B, embedding_dim)
        """
        with torch.no_grad():
            # Face features
            if self.face_encoder is not None:
                f_face = self.face_encoder(images)
            else:
                f_face = torch.zeros(images.size(0), 512, device=self.device)
            
            # CLIP visual features
            if self.clip_model is not None:
                # Use half precision only on CUDA
                if self.device.type == 'cuda':
                    f_clip = self.clip_model.visual(img_clip.half()).float()
                else:
                    f_clip = self.clip_model.visual(img_clip.float()).float()
            else:
                f_clip = torch.zeros(img_clip.size(0), 512, device=self.device)
            
            # Image feature fusion
            f_concat = torch.cat((f_face, f_clip), dim=1)
            z = self.img_fusion_projector(f_concat)
            z = F.normalize(z, dim=-1)
            
            # If texts provided, use text-fused embeddings
            if texts is not None and self.text_encoder is not None:
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                t = self.text_encoder(text_tokens).float()
                
                t_concat = torch.cat((f_concat, t), dim=1)
                t = self.text_fusion_projector(t_concat)
                t = F.normalize(t, dim=-1)
                return t
            
            return z
    
    def compute_similarity(
        self,
        query_embedding: torch.Tensor,
        gallery_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and gallery embeddings.
        
        Args:
            query_embedding: Query embedding (1, embedding_dim)
            gallery_embeddings: Gallery embeddings (N, embedding_dim)
        
        Returns:
            Similarity scores (1, N)
        """
        # Both should already be normalized
        similarity = torch.mm(query_embedding, gallery_embeddings.t())
        return similarity
    
    def load_weights(self, weights_path: str, strict: bool = False):
        """
        Load model weights.
        
        Args:
            weights_path: Path to weights file
            strict: Whether to strictly enforce state_dict keys match
        """
        if not weights_path:
            print("[CrossModalFaceMatcher] No weights path provided, using pretrained components.")
            return
        
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            
            # Handle different state_dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load weights
            missing, unexpected = self.load_state_dict(new_state_dict, strict=strict)
            
            if missing:
                print(f"[CrossModalFaceMatcher] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[CrossModalFaceMatcher] Unexpected keys: {len(unexpected)}")
            
            print(f"[CrossModalFaceMatcher] Loaded weights from {weights_path}")
        except Exception as e:
            print(f"[CrossModalFaceMatcher] Error loading weights: {e}")


class ImagePreprocessor:
    """
    Image preprocessor for cross-modal face matching.
    Uses PIL and torchvision.transforms for preprocessing.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize preprocessor.
        
        Args:
            device: Device for tensor operations
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Import torchvision transforms
        from torchvision import transforms
        
        # Transform for FaceNet (160x160)
        self.facenet_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Transform for CLIP (224x224)
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def preprocess(
        self,
        image: Union[str, Image.Image, np.ndarray],
        return_both: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_both: Whether to return both FaceNet and CLIP transforms
        
        Returns:
            Preprocessed tensor(s)
        """
        # Convert to PIL Image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Assume BGR format from OpenCV
            pil_image = Image.fromarray(image[:, :, ::-1])
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        facenet_tensor = self.facenet_transform(pil_image).unsqueeze(0).to(self.device)
        
        if return_both:
            clip_tensor = self.clip_transform(pil_image).unsqueeze(0).to(self.device)
            return facenet_tensor, clip_tensor
        
        return facenet_tensor
    
    def preprocess_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        return_both: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess batch of images.
        
        Args:
            images: List of input images
            return_both: Whether to return both FaceNet and CLIP transforms
        
        Returns:
            Batch of preprocessed tensors
        """
        facenet_tensors = []
        clip_tensors = []
        
        for img in images:
            if return_both:
                fn_tensor, clip_tensor = self.preprocess(img, return_both=True)
                facenet_tensors.append(fn_tensor)
                clip_tensors.append(clip_tensor)
            else:
                fn_tensor = self.preprocess(img, return_both=False)
                facenet_tensors.append(fn_tensor)
        
        facenet_batch = torch.cat(facenet_tensors, dim=0)
        
        if return_both:
            clip_batch = torch.cat(clip_tensors, dim=0)
            return facenet_batch, clip_batch
        
        return facenet_batch


# Import numpy for type hints
import numpy as np


def create_matcher(
    device: str = 'cuda',
    embedding_dim: int = 512,
    weights_path: Optional[str] = None
) -> CrossModalFaceMatcher:
    """
    Create a CrossModalFaceMatcher instance.
    
    Args:
        device: Device to run model
        embedding_dim: Dimension of embeddings
        weights_path: Path to pretrained weights
    
    Returns:
        CrossModalFaceMatcher instance
    """
    model = CrossModalFaceMatcher(
        device=device,
        embedding_dim=embedding_dim
    )
    
    if weights_path:
        model.load_weights(weights_path)
    
    model.to(model.device)
    model.eval()
    
    return model
