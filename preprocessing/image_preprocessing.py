"""Shared preprocessing pipelines for CNN training, evaluation, and inference.

The deterministic evaluation pipeline preserves the resize, crop, interpolation,
and normalization settings recorded with the existing checkpoints.
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

INPUT_SIZE = (224, 224)
CROP_SIZE = 224
RESIZE_SIZE = 256
INTERPOLATION = InterpolationMode.BILINEAR

# These values match checkpoints/preprocessing_config.json.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_training_transform():
    """Build the random augmentation, tensor conversion, and normalization pipeline.

    Use this transform only for training samples. Validation, test, and inference
    inputs should use :func:`create_evaluation_transform`.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            CROP_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
            interpolation=INTERPOLATION,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_evaluation_transform():
    """Build the deterministic resize, center-crop, and normalization pipeline.

    This same transform is used for validation, testing, and inference; it does
    not contain random augmentations.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=INTERPOLATION),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
