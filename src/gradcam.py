import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Heatmap Grad-CAM kustom untuk interpretasi keputusan model.

    Args:
        model (torch.nn.Module): Model yang divisualisasi.
        target_layer (torch.nn.Module): Lapisan konvolusi target heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Simpan aktivasi lapisan target saat forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Simpan gradien lapisan target saat backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Hasilkan heatmap Grad-CAM ternormalisasi [0, 1].

        Args:
            input_tensor (torch.Tensor): Batch satu citra (1, C, H, W).
            target_class (int atau None): Kelas target; kelas prediksi bila None.

        Returns:
            numpy.ndarray: Heatmap 2D seukuran citra masukan.
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture gradients and activations"
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()

    def visualize(self, input_tensor, image, target_class=None, save_path=None):
        """Tampilkan panel asli, heatmap, dan overlay Grad-CAM.

        Args:
            input_tensor (torch.Tensor): Batch satu citra untuk model.
            image (PIL.Image atau numpy.ndarray): Citra untuk panel asli.
            target_class (int atau None): Kelas target heatmap.
            save_path (str atau None): Path PNG tujuan; tampilkan bila None.
        """
        cam = self.generate(input_tensor, target_class)

        if isinstance(image, Image.Image):
            image = np.array(image)

        _, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(image)
        axes[2].imshow(cam, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def get_gradcam_target_layer(model):
    """Cari lapisan konvolusi terakhir backbone untuk Grad-CAM.

    Args:
        model (torch.nn.Module): Model klasifikasi (DenseNet-121 + Attention).

    Returns:
        torch.nn.Module atau None: Lapisan target; None bila tak ditemukan.
    """
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        features = model.backbone.features
        if hasattr(features, "denseblock4"):
            return features.denseblock4
        elif hasattr(features, "norm5"):
            return features.norm5
    return None


def visualize_gradcam(model, dataloader, device, num_samples=5, save_dir="checkpoints"):
    """Hasilkan figure Grad-CAM beberapa sampel test ke direktori run.

    Args:
        model (torch.nn.Module): Model terlatih.
        dataloader (DataLoader): DataLoader test.
        device (torch.device): CPU atau CUDA.
        num_samples (int): Jumlah sampel divisualisasi.
        save_dir (str): Direktori keluaran figure.
    """
    model.eval()
    target_layer = get_gradcam_target_layer(model)

    if target_layer is None:
        print("Could not find target layer for Grad-CAM")
        return

    gradcam = GradCAM(model, target_layer)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    _, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    count = 0
    for images, labels in dataloader:
        for sample_index in range(min(images.size(0), num_samples - count)):
            img_tensor = images[sample_index : sample_index + 1].to(device)
            label = labels[sample_index].item()

            with torch.no_grad():
                output = model(img_tensor)
                prediction = (torch.sigmoid(output) > 0.5).long().item()

            cam = gradcam.generate(img_tensor)

            img_display = images[sample_index].cpu() * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            axes[count, 0].imshow(img_display)
            axes[count, 0].set_title(f"Original\nTrue: {'PCOS' if label else 'Normal'}")
            axes[count, 0].axis("off")

            axes[count, 1].imshow(cam, cmap="jet")
            axes[count, 1].set_title(
                f"Grad-CAM\nPred: {'PCOS' if prediction else 'Normal'}"
            )
            axes[count, 1].axis("off")

            axes[count, 2].imshow(img_display)
            axes[count, 2].imshow(cam, cmap="jet", alpha=0.5)
            axes[count, 2].set_title("Overlay")
            axes[count, 2].axis("off")

            count += 1
            if count >= num_samples:
                break

        if count >= num_samples:
            break

    plt.tight_layout()
    plt.savefig(f"{save_dir}/gradcam_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grad-CAM saved to {save_dir}/gradcam_visualization.png")
