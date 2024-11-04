import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

dataset = datasets.CIFAR10(root='./data', train=False, download=True)
image, label = dataset[16]  # Original image of a dog in PIL

# Define transformations
random_rotation = transforms.RandomRotation(degrees=45)
random_horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
random_adjust_sharpness = transforms.RandomAdjustSharpness(sharpness_factor=2)
to_tensor = transforms.ToTensor()  # Convert to tensor
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
random_erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3))

# Apply transformations
rotated_image = random_rotation(image)
flipped_image = random_horizontal_flip(image)
jittered_image = color_jitter(image)
sharpened_image = random_adjust_sharpness(image)
tensor_image = to_tensor(image)
erased_image = random_erasing(tensor_image.clone())


def show_image(img, title, is_tensor=False):
    plt.imshow(img)
    plt.title(title)
    plt.show()


show_image(erased_image, 'Random Erased Image', is_tensor=True)
show_image(rotated_image, 'Random Rotation')
show_image(flipped_image, 'Random Horizontal Flip')
show_image(jittered_image, 'Color Jitter')
show_image(sharpened_image, 'Random Adjust Sharpness')
