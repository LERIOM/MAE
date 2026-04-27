from __future__ import annotations

import json
import random
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16

from src.model import MAE, MAEDecoder


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEMO_DIR = PROJECT_ROOT / "demo"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

MAE_RECONSTRUCTION_CHECKPOINT = DEMO_DIR / "best_mae_flower.pth"
MAE_CLASSIFIER_CHECKPOINT = CHECKPOINT_DIR / "mae_classifier_best.pth"
RESNET_CLASSIFIER_CHECKPOINT = CHECKPOINT_DIR / "resnet_classifier_best.pth"
FLOWER_NAMES_PATH = DATA_DIR / "flowers" / "cat_to_name.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchvisionViTMAEEncoder(nn.Module):
    def __init__(self, img_size: int, patch_size: int = 16, in_chans: int = 3, mask_ratio: float = 0.0):
        super().__init__()
        if patch_size != 16:
            raise ValueError("torchvision ViT-B/16 requires patch_size=16")
        if in_chans != 3:
            raise ValueError("torchvision ViT-B/16 requires RGB images")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.vit = vit_b_16(weights=None, image_size=img_size)
        self.encoder_dim = self.vit.hidden_dim

    def random_masking(self, x: torch.Tensor):
        batch_size, num_patches, dim = x.shape
        num_keep = int(num_patches * (1 - self.mask_ratio))
        if num_keep <= 0:
            raise ValueError(f"mask_ratio={self.mask_ratio} keeps no visible patches")
        if num_keep == num_patches:
            ids = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch_size, -1)
            mask = torch.zeros(batch_size, num_patches, device=x.device)
            return x, mask, ids, ids

        noise = torch.rand(batch_size, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        x_vis = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim),
        )

        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_vis, mask, ids_restore, ids_keep

    def forward(self, images: torch.Tensor):
        x = self.vit._process_input(images)
        patch_pos_embed = self.vit.encoder.pos_embedding[:, 1:, :]
        x = x + patch_pos_embed

        x_vis, mask, ids_restore, ids_keep = self.random_masking(x)

        cls_token = self.vit.class_token + self.vit.encoder.pos_embedding[:, :1, :]
        cls_token = cls_token.expand(images.shape[0], -1, -1)
        x_vis = torch.cat([cls_token, x_vis], dim=1)
        x_vis = self.vit.encoder.dropout(x_vis)
        x_vis = self.vit.encoder.layers(x_vis)
        x_vis = self.vit.encoder.ln(x_vis)
        return x_vis[:, 1:, :], mask, ids_restore, ids_keep


class MAEClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes),
        )

    def forward(self, images: torch.Tensor):
        latent, _, _, _ = self.encoder(images)
        features = latent.mean(dim=1)
        return self.classifier(features)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def count_module_list_depth(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    indices = set()
    for key in state_dict:
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            index_text = suffix.split(".", 1)[0]
            if index_text.isdigit():
                indices.add(int(index_text))

    if not indices:
        raise ValueError(f"Could not infer module depth from keys starting with {prefix!r}")
    return max(indices) + 1


def build_mae_from_state_dict(state_dict: dict[str, torch.Tensor], mask_ratio: float):
    patch_weight = state_dict["encoder.vit.conv_proj.weight"]
    encoder_dim, in_chans, patch_h, patch_w = patch_weight.shape
    if patch_h != patch_w:
        raise ValueError(f"Expected square patches, got {patch_h}x{patch_w}")

    pos_embed = state_dict["encoder.vit.encoder.pos_embedding"]
    num_patches = pos_embed.shape[1] - 1
    grid_size = int(num_patches**0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches={num_patches} does not form a square grid")

    img_size = grid_size * patch_h
    decoder_dim = state_dict["decoder.pos_embed"].shape[2]
    decoder_depth = count_module_list_depth(state_dict, "decoder.blocks.")

    encoder = TorchvisionViTMAEEncoder(
        img_size=img_size,
        patch_size=patch_h,
        in_chans=in_chans,
        mask_ratio=mask_ratio,
    )
    decoder = MAEDecoder(
        num_patches=num_patches,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        patch_size=patch_h,
        in_chans=in_chans,
        decoder_depth=decoder_depth,
        decoder_heads=8,
    )
    model = MAE(encoder, decoder, norm_pix_loss=True)
    model.load_state_dict(state_dict, strict=True)

    config = {
        "img_size": img_size,
        "patch_size": patch_h,
        "num_patches": num_patches,
        "encoder_dim": encoder_dim,
        "decoder_dim": decoder_dim,
        "decoder_depth": decoder_depth,
    }
    return model, config


@st.cache_resource(show_spinner=False)
def load_reconstruction_model(checkpoint_path: str):
    state_dict = torch_load(Path(checkpoint_path))
    model, config = build_mae_from_state_dict(state_dict, mask_ratio=0.75)
    model.to(DEVICE)
    model.eval()
    return model, config


@st.cache_resource(show_spinner=False)
def load_mae_classifier(checkpoint_path: str):
    checkpoint = torch_load(Path(checkpoint_path))
    config = checkpoint["encoder_config"]
    class_names = checkpoint["class_names"]

    encoder = TorchvisionViTMAEEncoder(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_chans=config["in_chans"],
        mask_ratio=0.0,
    )
    model = MAEClassifier(
        encoder=encoder,
        encoder_dim=config["encoder_dim"],
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, class_names, checkpoint


@st.cache_resource(show_spinner=False)
def load_resnet_classifier(checkpoint_path: str):
    checkpoint = torch_load(Path(checkpoint_path))
    class_names = checkpoint["class_names"]
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, class_names, checkpoint


@st.cache_data(show_spinner=False)
def load_flower_names() -> dict[str, str]:
    if not FLOWER_NAMES_PATH.exists():
        return {}
    return json.loads(FLOWER_NAMES_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def list_project_images(source: str) -> list[str]:
    if source == "Fleurs":
        roots = [DATA_DIR / "flowers" / "valid", DATA_DIR / "flowers" / "test"]
    else:
        roots = [DATA_DIR / "japanese_textiles", DATA_DIR / "sri_lankan_textiles"]

    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return [str(path) for path in sorted(paths)]


@st.cache_data(show_spinner=False)
def dataset_counts() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    flowers_root = DATA_DIR / "flowers"
    for split in ("train", "valid", "test"):
        split_dir = flowers_root / split
        if split_dir.exists():
            count = sum(1 for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            rows.append({"Dataset": "Fleurs", "Partition": split, "Images": count})

    for name in ("japanese_textiles", "sri_lankan_textiles"):
        root = DATA_DIR / name
        if root.exists():
            count = sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            rows.append({"Dataset": "Textiles", "Partition": name.replace("_", " "), "Images": count})

    return rows


def format_sample(path_text: str) -> str:
    path = Path(path_text)
    flower_names = load_flower_names()
    if "flowers" in path.parts:
        class_id = path.parent.name
        label = flower_names.get(class_id, f"classe {class_id}")
        return f"{label} - {path.name}"
    return path.name


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def resnet_image_to_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform(image).unsqueeze(0)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(image_tensor)


def build_masked_images(model: MAE, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    patches = model.patchify(images).clone()
    patches = patches * (1.0 - mask.unsqueeze(-1).to(dtype=patches.dtype))
    return model.unpatchify(patches)


def denormalize_predicted_patches(model: MAE, images: torch.Tensor, pred_patches: torch.Tensor) -> torch.Tensor:
    if not model.norm_pix_loss:
        return pred_patches

    target_patches = model.patchify(images)
    mean = target_patches.mean(dim=-1, keepdim=True)
    var = target_patches.var(dim=-1, keepdim=True)
    return pred_patches * (var + 1.0e-6).sqrt() + mean


def build_reconstructed_images(
    model: MAE,
    images: torch.Tensor,
    pred_patches: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_patches = denormalize_predicted_patches(model, images, pred_patches)
    original_patches = model.patchify(images)
    mask = mask.unsqueeze(-1).to(dtype=pred_patches.dtype)
    blended_patches = original_patches * (1.0 - mask) + pred_patches * mask
    return model.unpatchify(blended_patches)


def mask_to_pil(mask: torch.Tensor, grid_size: int, image_size: int) -> Image.Image:
    mask_image = mask.detach().cpu().reshape(grid_size, grid_size).mul(255).byte().numpy()
    return Image.fromarray(mask_image, mode="L").resize((image_size, image_size), Image.Resampling.NEAREST)


def run_reconstruction(model: MAE, config: dict[str, int], image: Image.Image, mask_ratio: float, seed: int):
    image_tensor = image_to_tensor(image, config["img_size"]).to(DEVICE)
    previous_mask_ratio = model.encoder.mask_ratio
    model.encoder.mask_ratio = mask_ratio

    try:
        torch.manual_seed(seed)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            loss, _, pred_patches, mask, _, _ = model(
                image_tensor,
                return_aux=True,
                return_loss=True,
            )
            masked = build_masked_images(model, image_tensor, mask)
            reconstructed = build_reconstructed_images(model, image_tensor, pred_patches, mask)
    finally:
        model.encoder.mask_ratio = previous_mask_ratio

    grid_size = int(config["num_patches"] ** 0.5)
    return {
        "loss": loss.item(),
        "original": tensor_to_pil(image_tensor[0]),
        "masked": tensor_to_pil(masked[0]),
        "reconstructed": tensor_to_pil(reconstructed[0]),
        "mask": mask_to_pil(mask[0], grid_size, config["img_size"]),
        "masked_patches": int(mask.sum().item()),
        "total_patches": int(mask.numel()),
    }


def predict_topk(model: nn.Module, tensor: torch.Tensor, class_names: list[str], k: int = 5):
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(k, len(class_names)))

    flower_names = load_flower_names()
    rows = []
    for rank, (value, index) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist()), start=1):
        class_id = class_names[index]
        flower_name = flower_names.get(class_id, f"classe {class_id}")
        rows.append({
            "Rang": rank,
            "Classe": flower_name,
            "ID": class_id,
            "Confiance": f"{value * 100:.2f}%",
        })
    return rows


def parse_training_report(report_path: Path):
    if not report_path.exists():
        return None

    text = report_path.read_text(encoding="utf-8")
    best_match = re.search(r"Best val_acc=([0-9.]+)%", text)
    epochs = re.findall(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+train_loss=([0-9.]+)\s+\|\s+train_acc=([0-9.]+)%\s+\|\s+"
        r"val_loss=([0-9.]+)\s+\|\s+val_acc=([0-9.]+)%",
        text,
    )
    return {
        "best_val_acc": float(best_match.group(1)) if best_match else None,
        "epochs": epochs,
    }


def show_prediction_table(title: str, rows: list[dict[str, str | int]]):
    st.subheader(title)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def selected_input_image(source_kind: str, sample_domain: str) -> tuple[Image.Image | None, str]:
    if source_kind == "Importer":
        uploaded_file = st.sidebar.file_uploader("Image", type=sorted(ext.strip(".") for ext in IMAGE_EXTENSIONS))
        if uploaded_file is None:
            sample_paths = list_project_images("Fleurs")
            if not sample_paths:
                return None, "Aucune image disponible"
            path = Path(sample_paths[0])
            return open_rgb_image(path), format_sample(str(path))
        return Image.open(uploaded_file).convert("RGB"), uploaded_file.name

    sample_paths = list_project_images(sample_domain)
    if not sample_paths:
        return None, f"Aucune image {sample_domain.lower()} trouvée"

    index = st.sidebar.selectbox(
        "Échantillon",
        range(len(sample_paths)),
        format_func=lambda idx: format_sample(sample_paths[idx]),
    )
    path = Path(sample_paths[index])
    return open_rgb_image(path), format_sample(str(path))


def main():
    st.set_page_config(page_title="MAE Studio", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
        [data-testid="stMetric"] {background: rgba(15, 23, 42, 0.04); border: 1px solid rgba(15, 23, 42, 0.08); padding: 0.75rem; border-radius: 8px;}
        [data-testid="stMetricValue"] {font-size: 1.45rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 0.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("MAE Studio")
    source_kind = st.sidebar.radio("Image", ("Exemple", "Importer"), horizontal=True)
    sample_domain = st.sidebar.radio("Source", ("Fleurs", "Textiles"), horizontal=True, disabled=source_kind == "Importer")
    mask_ratio = st.sidebar.slider("Masque", min_value=0.10, max_value=0.95, value=0.75, step=0.05)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=999_999, value=42, step=1)

    checkpoint_path = MAE_RECONSTRUCTION_CHECKPOINT
    if not checkpoint_path.exists():
        st.error(f"Checkpoint MAE introuvable: {checkpoint_path}")
        return

    image, image_label = selected_input_image(source_kind, sample_domain)
    if image is None:
        st.error(image_label)
        return

    st.title("MAE Studio")
    st.caption("Projet UQAC - Masked Autoencoder, classification fleurs et comparaison ResNet.")

    with st.spinner("Chargement du MAE"):
        reconstruction_model, reconstruction_config = load_reconstruction_model(str(checkpoint_path))

    tabs = st.tabs(["Reconstruction", "Classification", "Résultats", "Données"])

    with tabs[0]:
        result = run_reconstruction(
            reconstruction_model,
            reconstruction_config,
            image,
            mask_ratio=float(mask_ratio),
            seed=int(seed),
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Loss MAE", f"{result['loss']:.4f}")
        metric_cols[1].metric("Patches masqués", f"{result['masked_patches']}/{result['total_patches']}")
        metric_cols[2].metric("Image", f"{reconstruction_config['img_size']} px")
        metric_cols[3].metric("Device", DEVICE.type.upper())

        image_cols = st.columns(4)
        image_cols[0].image(result["original"], caption=f"Original - {image_label}", use_container_width=True)
        image_cols[1].image(result["masked"], caption=f"Masquée à {mask_ratio:.0%}", use_container_width=True)
        image_cols[2].image(result["reconstructed"], caption="Reconstruite", use_container_width=True)
        image_cols[3].image(result["mask"], caption="Carte du masque", use_container_width=True)

    with tabs[1]:
        top_cols = st.columns(3)
        mae_report = parse_training_report(DEMO_DIR / "report_mae.txt")
        resnet_report = parse_training_report(DEMO_DIR / "report_resnet.txt")
        mae_best = mae_report["best_val_acc"] if mae_report else None
        resnet_best = resnet_report["best_val_acc"] if resnet_report else None
        top_cols[0].metric("MAE classifier", f"{mae_best:.2f}%" if mae_best is not None else "n/a")
        top_cols[1].metric("ResNet18", f"{resnet_best:.2f}%" if resnet_best is not None else "n/a")
        if mae_best is not None and resnet_best is not None:
            top_cols[2].metric("Écart", f"{mae_best - resnet_best:+.2f} pts")
        else:
            top_cols[2].metric("Écart", "n/a")

        if st.button("Comparer les prédictions", use_container_width=True):
            if not MAE_CLASSIFIER_CHECKPOINT.exists() or not RESNET_CLASSIFIER_CHECKPOINT.exists():
                st.error("Checkpoints de classification introuvables dans outputs/checkpoints.")
            else:
                with st.spinner("Inférence des classifieurs"):
                    mae_classifier, mae_classes, _ = load_mae_classifier(str(MAE_CLASSIFIER_CHECKPOINT))
                    resnet_classifier, resnet_classes, _ = load_resnet_classifier(str(RESNET_CLASSIFIER_CHECKPOINT))
                    mae_tensor = image_to_tensor(image, 256)
                    resnet_tensor = resnet_image_to_tensor(image)
                    mae_predictions = predict_topk(mae_classifier, mae_tensor, mae_classes)
                    resnet_predictions = predict_topk(resnet_classifier, resnet_tensor, resnet_classes)

                pred_cols = st.columns(2)
                with pred_cols[0]:
                    show_prediction_table("Encodeur MAE fine-tuné", mae_predictions)
                with pred_cols[1]:
                    show_prediction_table("ResNet18 ImageNet", resnet_predictions)

    with tabs[2]:
        metrics = st.columns(3)
        metrics[0].metric("Pré-entraînement", "200 epochs")
        metrics[1].metric("Classes fleurs", "102")
        metrics[2].metric("Checkpoint MAE", "361 Mo")

        curve_cols = st.columns(2)
        with curve_cols[0]:
            show_image_if_exists(DEMO_DIR / "mae_classifier_curves.png", "Courbes du classifieur MAE")
        with curve_cols[1]:
            show_image_if_exists(DEMO_DIR / "resnet_classifier_curves.png", "Courbes ResNet18")

        show_image_if_exists(DEMO_DIR / "pretrained_loss_curve.png", "Loss du pré-entraînement MAE")

        st.subheader("Évolution des reconstructions")
        gallery_candidates = [
            ("Epoch 0", DEMO_DIR / "flower_0.png"),
            ("Epoch 50", DEMO_DIR / "flower_50.png"),
            ("Epoch 100", DEMO_DIR / "flower_100.png"),
            ("Epoch 200", DEMO_DIR / "flower_200.png"),
        ]
        gallery_cols = st.columns(4)
        for col, (caption, path) in zip(gallery_cols, gallery_candidates):
            with col:
                show_image_if_exists(path, caption)

        sweep_path = OUTPUT_DIR / "pretrained_visualisations" / "flowers" / "epoch_200_mask_ratio_sweep_img_00.png"
        if sweep_path.exists():
            st.image(str(sweep_path), caption="Sensibilité au taux de masque", use_container_width=True)

    with tabs[3]:
        rows = dataset_counts()
        total_images = sum(int(row["Images"]) for row in rows)
        data_cols = st.columns(3)
        data_cols[0].metric("Images indexées", f"{total_images:,}".replace(",", " "))
        data_cols[1].metric("Partitions", str(len(rows)))
        data_cols[2].metric("Artefacts demo", str(len(list(DEMO_DIR.glob('*')))))
        st.dataframe(rows, use_container_width=True, hide_index=True)

        sample_cols = st.columns(3)
        sample_images = [
            DEMO_DIR / "textile_100.png",
            DEMO_DIR / "overfit_textile_500.png",
            DEMO_DIR / "flower_200_all.png",
        ]
        sample_captions = ["Textiles", "Overfit contrôlé", "Fleurs à 200 epochs"]
        for col, path, caption in zip(sample_cols, sample_images, sample_captions):
            with col:
                show_image_if_exists(path, caption)


if __name__ == "__main__":
    main()
