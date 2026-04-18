import torch
from torch import nn

"""
Encoder class

Uses a transformer to encode images

Each image is preprocessed in the patchify
Shape : (B, C, H, W) => (B, N_patches, patch_size² * C)
"""
class Encoder(nn.Module):
    def __init__(
            self,
            img_size=256,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            num_heads=8,
            mlp_dim=3072,
            num_layers=4,
            dropout=0.1,
            ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def patchify(self, images, visible_indices):
        B, C, H, W = images.shape

        P = 16 # patch size
        patches = images.reshape(B, C, H//P, P, W//P, P) # H = H//P * P ; W = W//P * P
        patches = patches.permute(0, 2, 4, 1, 3, 5) # (B, H//P, W//P, C, P, P)
        patches = patches.reshape(B, -1, C * P * P)

        patches = patches[:, visible_indices, :] # keep visible patches

        return patches
    
    def forward(self, x):
        masked_images, visible_indices = x
        patches = self.patchify(masked_images, visible_indices)

        tokens = self.patch_embedding(patches)
        tokens = tokens + self.pos_embedding[:, visible_indices, :]

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        return tokens, visible_indices

"""
Decoder class

Uses a CNN to decode images
"""
class Decoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        out_channels=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)

        # 4 layers : 16 => 32 => 64 => 128 => 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        tokens, visible_indices = x
        B = tokens.shape[0]

        full_tokens = self.mask_token.expand(B, self.num_patches, -1).clone()
        full_tokens[:, visible_indices, :] = tokens

        full_tokens = full_tokens + self.pos_embedding
        full_tokens = self.proj(full_tokens)

        full_tokens = full_tokens.permute(0, 2, 1)
        full_tokens = full_tokens.reshape(B, self.embed_dim, self.grid_size, self.grid_size)

        output = self.decoder(full_tokens)

        return output

"""
AutoEncoder class

Uses Encoder and Decoder
"""
class AutoEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=8,
        mlp_dim=3072,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=in_channels,
            embed_dim=embed_dim,
        )

    def forward(self, x):
        tokens, visible_indices = self.encoder(x)
        output = self.decoder((tokens, visible_indices))
        return output