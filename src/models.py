import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from config import Config

class ConvStem(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.stem(x)

class ImageCoordinateEncoding(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height)
        x_coords = torch.linspace(-1, 1, width)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack coordinates
        self.coords = torch.stack([x_grid, y_grid], dim=0)
    
    def forward(self, x):
        # Add coordinate channels to input
        coords = self.coords.to(x.device)
        coords = coords.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        return torch.cat([x, coords], dim=1)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads, patch_size, use_coords=False):
        super().__init__()
        self.patch_size = patch_size
        self.use_coords = use_coords
        # 256 channels from ConvStem + 2 coordinate channels if using coords
        in_features = patch_size[0] * (258 if use_coords else 256)
        self.patch_embed = nn.Linear(in_features, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Reshape into patches
        B, C, H, W = x.shape
        
        # Ensure height is divisible by patch height
        if H % self.patch_size[0] != 0:
            pad_h = self.patch_size[0] - (H % self.patch_size[0])
            x = F.pad(x, (0, 0, 0, pad_h))
            H = H + pad_h
        
        # Reshape into patches
        x = rearrange(x, 'b c (h p1) w -> b (w h) (p1 c)',
                     p1=self.patch_size[0])
        
        # Apply patch embedding
        x = self.patch_embed(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x

class TeacherNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = ConvStem()
        self.coord_encoding = ImageCoordinateEncoding(
            Config.SPECTROGRAM_SIZE[0],
            Config.SPECTROGRAM_SIZE[1]
        )
        self.vit_encoder = ViTEncoder(
            dim=Config.EMBED_DIM,
            depth=Config.TEACHER_DEPTH,
            num_heads=Config.TEACHER_HEADS,
            patch_size=Config.PATCH_SIZE,
            use_coords=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM // 2),
            nn.GELU(),
            nn.Linear(Config.EMBED_DIM // 2, len(Config.EMOTION_LABELS))
        )
    
    def forward(self, x):
        # Apply conv stem
        x = self.conv_stem(x)
        
        # Add coordinate encoding
        x = self.coord_encoding(x)
        
        # Apply ViT encoder
        x = self.vit_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classify
        x = self.classifier(x)
        return x

class StudentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = ConvStem()
        self.vit_encoder = ViTEncoder(
            dim=Config.EMBED_DIM,
            depth=Config.STUDENT_DEPTH,
            num_heads=Config.STUDENT_HEADS,
            patch_size=Config.PATCH_SIZE,
            use_coords=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM // 2),
            nn.GELU(),
            nn.Linear(Config.EMBED_DIM // 2, len(Config.EMOTION_LABELS))
        )
    
    def forward(self, x):
        # Apply conv stem
        x = self.conv_stem(x)
        
        # Apply ViT encoder
        x = self.vit_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classify
        x = self.classifier(x)
        return x 