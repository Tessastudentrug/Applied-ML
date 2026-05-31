from torchvision import transforms


def get_train_transform(image_size):
    """Used for training: Includes random augmentations to stop overfitting."""
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
    """Used for Validation/Testing: NO augmentation, just clean images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
