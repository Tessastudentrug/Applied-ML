"""
Image preprocessing pipelines for training and evaluation.

Defines torchvision transforms for:
- Training (with augmentation)
- Evaluation / inference (deterministic)
"""

from torchvision import transforms


def get_train_transform(image_size):
    """
    Training image transform pipeline with data augmentation.

    Adds randomness to improve generalization and reduce overfitting.

    Args:
        image_size: Target image size (H, W).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip the face
            transforms.RandomRotation(degrees=10),  # Mild rotation (max 10 degrees)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def get_eval_transform(image_size):
    """
    Evaluation / inference image transform pipeline.

    Uses deterministic preprocessing only (no augmentation).

    Args:
        image_size: Target image size (H, W).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
