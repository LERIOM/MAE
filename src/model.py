import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: [B, 3, 256, 256]
        return: [B, 256, embed_dim]
        """
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dim, dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        mlp_ratio=4.0,
        mask_ratio=0.5,
        dropout=0.0,
    ):
        super().__init__()

        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_dim
        )

        self.num_patches = self.patch_embed.num_patches
        self.encoder_dim = encoder_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=encoder_dim,
                num_heads=encoder_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(encoder_depth)
        ])
        self.norm = nn.LayerNorm(encoder_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def random_masking(self, x):
        """
        x: [B, N, D]

        returns:
            x_vis: [B, N_keep, D]
            mask: [B, N] avec 0=visible, 1=masqué
            ids_restore: [B, N]
            ids_keep: [B, N_keep]
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]

        x_vis = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_vis, mask, ids_restore, ids_keep

    def forward(self, imgs):
        """
        imgs: [B, 3, 256, 256]

        returns:
            latent: [B, N_keep, encoder_dim]
            mask: [B, 256]
            ids_restore: [B, 256]
            ids_keep: [B, N_keep]
        """
        x = self.patch_embed(imgs)
        x = x + self.pos_embed

        x_vis, mask, ids_restore, ids_keep = self.random_masking(x)

        for blk in self.blocks:
            x_vis = blk(x_vis)

        latent = self.norm(x_vis)

        return latent, mask, ids_restore, ids_keep
    
class MAEDecoder(nn.Module):
    def __init__(
        self,
        num_patches=256,
        encoder_dim=512,
        decoder_dim=512,
        patch_size=16,
        in_chans=3,
        decoder_depth=6,
        decoder_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.in_chans = in_chans

        if encoder_dim == decoder_dim:
            self.decoder_embed = nn.Identity()
        else:
            self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_dim,
                num_heads=decoder_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        self.decoder_pred = nn.Linear(decoder_dim, in_chans * patch_size * patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class MAE(nn.Module):
    def __init__(self, encoder, decoder, norm_pix_loss=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.norm_pix_loss = norm_pix_loss

    def decode_patches(self, latent, ids_keep):
        latent = self.decoder.decoder_embed(latent)

        B, _, D = latent.shape
        N = self.decoder.pos_embed.shape[1]
        dtype = latent.dtype

        latent_full = self.decoder.mask_token.to(dtype=dtype).expand(B, N, D).clone()
        latent_full.scatter_(
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
            src=latent,
        )

        latent_full = latent_full + self.decoder.pos_embed.to(dtype=dtype)

        for blk in self.decoder.blocks:
            latent_full = blk(latent_full)

        pred = self.decoder.norm(latent_full)
        pred = self.decoder.decoder_pred(pred)
        return pred

    def patchify(self, imgs):
        """
        imgs: [B, C, H, W]
        return: [B, N, C * P * P]
        """
        B, C, H, W = imgs.shape
        p = self.encoder.patch_size
        expected_c = self.encoder.in_chans
        expected_patches = self.encoder.num_patches

        if C != expected_c:
            raise ValueError(f"expected {expected_c} channels, got {C}")
        if H != W:
            raise ValueError(f"expected square images, got H={H}, W={W}")
        if H % p != 0:
            raise ValueError(f"image size ({H}) must be divisible by patch_size ({p})")

        grid_size = H // p
        num_patches = grid_size * grid_size
        if num_patches != expected_patches:
            raise ValueError(
                f"expected {expected_patches} patches, got {num_patches}; "
                f"check the input image size"
            )

        patches = imgs.reshape(B, C, grid_size, p, grid_size, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(B, num_patches, C * p * p)
        return patches

    def unpatchify(self, patches):
        """
        patches: [B, N, C * P * P]
        return: [B, C, H, W]
        """
        B, N, patch_dim = patches.shape
        p = self.decoder.patch_size
        c = self.decoder.in_chans
        grid_size = int(N ** 0.5)
        expected_patch_dim = c * p * p

        if grid_size * grid_size != N:
            raise ValueError(f"num_patches ({N}) must form a square grid")
        if patch_dim != expected_patch_dim:
            raise ValueError(
                f"patch_dim ({patch_dim}) must match in_chans * patch_size^2 ({expected_patch_dim})"
            )

        imgs = patches.reshape(B, grid_size, grid_size, c, p, p)
        imgs = imgs.permute(0, 3, 1, 4, 2, 5).contiguous()
        imgs = imgs.reshape(B, c, grid_size * p, grid_size * p)
        return imgs

    def forward_loss(self, imgs, pred_patches, mask):
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        loss = (pred_patches - target) ** 2
        loss = loss.mean(dim=-1)

        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            raise ValueError("mask must contain at least one masked patch")

        loss = (loss * mask).sum() / mask_sum
        return loss

    def forward(self, imgs, return_aux=False, return_loss=False):
        latent, mask, ids_restore, ids_keep = self.encoder(imgs)
        pred_patches = self.decode_patches(latent, ids_keep)
        reconstruction = self.unpatchify(pred_patches)
        loss = None

        if return_loss:
            loss = self.forward_loss(imgs, pred_patches, mask)

        if return_aux:
            if return_loss:
                return loss, reconstruction, pred_patches, mask, ids_restore, ids_keep
            return reconstruction, pred_patches, mask, ids_restore, ids_keep

        if return_loss:
            return loss, reconstruction
        return reconstruction

def build_mae(
    img_size=256,
    patch_size=16,
    in_chans=3,
    encoder_dim=512,
    encoder_depth=6,
    encoder_heads=8,
    decoder_dim=512,
    decoder_depth=6,
    decoder_heads=8,
    mlp_ratio=4.0,
    mask_ratio=0.5,
    dropout=0.0,
    norm_pix_loss=True,
):
    encoder = MAEEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        encoder_dim=encoder_dim,
        encoder_depth=encoder_depth,
        encoder_heads=encoder_heads,
        mlp_ratio=mlp_ratio,
        mask_ratio=mask_ratio,
        dropout=dropout,
    )

    decoder = MAEDecoder(
        num_patches=encoder.num_patches,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        patch_size=encoder.patch_size,
        in_chans=encoder.in_chans,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )

    model = MAE(encoder, decoder, norm_pix_loss=norm_pix_loss)
    return model
