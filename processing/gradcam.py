import torch
from matplotlib import pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import wandb
from torchvision import transforms


def unnormalize(image, mean, std):
    """Reverses the normalization process on an image tensor"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean
    return image


def run_gradcam(model, testloader, device, classes):
    grad_cam = GradCAM(model, target_layer=model.layer4)

    # Sample image
    model.eval()
    images, labels = next(iter(testloader))
    image = images[0].unsqueeze(0).to(device)

    output = model(image)
    predicted_class_idx = output.argmax(dim=1).item()
    true_class_idx = labels[0].item()

    predicted_class_name = classes[predicted_class_idx]
    true_class_name = classes[true_class_idx]

    activation_map = grad_cam(predicted_class_idx, output)

    activation_map = activation_map[0].cpu()

    heatmap = to_pil_image(activation_map, mode='F')

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    unnormalized_image = unnormalize(images[0].cpu(), mean, std)

    original_image_pil = to_pil_image(unnormalized_image)

    # Overlay heatmap on the original image
    result = overlay_mask(original_image_pil, heatmap, alpha=0.5)

    # Save original and Grad-CAM image
    original_image_pil.save("original_image.png")
    result.save("gradcam_result.png")

    wandb.log({
        "Original Image": wandb.Image(original_image_pil, caption=f"True class: {true_class_name}"),
        "Grad-CAM Image": wandb.Image(result, caption=f"Predicted class: {predicted_class_name}")
    })
