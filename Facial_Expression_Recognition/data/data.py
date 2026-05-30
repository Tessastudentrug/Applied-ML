import os

import kagglehub
import pandas as pd
import py7zr
import torch
from features.ExpW_preprocessor import ExpWPreprocessor
from features.preprocessing import get_eval_transform, get_train_transform
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

DATASET_ID = "shahzadabbas/expression-in-the-wild-expw-dataset"


class FERImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.csv_path = os.path.join(dataset_dir, "Stratified_10k_Metadata.csv")
        self.image_dir = os.path.join(dataset_dir, "Stratified_10k_Cleaned_64x64")

        self.df = pd.read_csv(self.csv_path)

        self.file_paths = [
            os.path.join(self.image_dir, img_name) for img_name in self.df["image_name"]
        ]

        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("L")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataset_dir():
    dataset_dir = kagglehub.dataset_download(DATASET_ID)

    extracted_dir = os.path.join(dataset_dir, "origin")

    if os.path.isdir(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
        return dataset_dir

    old_archive_part = os.path.join(dataset_dir, "origin.7z.001.001")
    new_archive_part = os.path.join(dataset_dir, "origin.7z.001")

    if os.path.exists(old_archive_part) and not os.path.exists(new_archive_part):
        os.rename(old_archive_part, new_archive_part)

    archive_parts = sorted(
        os.path.join(dataset_dir, filename)
        for filename in os.listdir(dataset_dir)
        if filename.startswith("origin.7z.")
    )

    combined_archive = os.path.join(dataset_dir, "combined.7z")

    if not os.path.exists(combined_archive):
        print("Combining archive parts...")

        with open(combined_archive, "wb") as outfile:
            for part in archive_parts:
                print(f"Adding {os.path.basename(part)}")

                with open(part, "rb") as infile:
                    outfile.write(infile.read())

    print("Extracting dataset...")

    with py7zr.SevenZipFile(combined_archive, "r") as archive:
        archive.extractall(path=dataset_dir)

    print("Extraction complete.")

    return dataset_dir


def get_dataloaders(batch_size=32, image_size=64, train_split=0.7, val_split=0.10):
    dataset_dir = get_dataset_dir()

    preprocessed_csv = os.path.join(dataset_dir, "Stratified_10k_Metadata.csv")
    preprocessed_dir = os.path.join(dataset_dir, "Stratified_10k_Cleaned_64x64")

    if not os.path.exists(preprocessed_csv) or not os.path.isdir(preprocessed_dir):
        preprocessor = ExpWPreprocessor(
            raw_metadata_path=os.path.join(dataset_dir, "label.lst"),
            stratified_csv_path=preprocessed_csv,
            input_dir_pattern=os.path.join(dataset_dir, "origin", "*.jpg"),
            output_dir=preprocessed_dir,
        )

        images = preprocessor.build_balanced_metadata(samples_per_class=1500)
        input_dir = preprocessor.find_input_dir()
        preprocessor.run(images, input_dir)

    # 1. Create TWO datasets: one with train augmentations, one with clean eval rules
    dataset_train = FERImageDataset(dataset_dir, transform=get_train_transform())
    dataset_eval = FERImageDataset(dataset_dir, transform=get_eval_transform())

    # 2. Calculate the split sizes
    total_size = len(dataset_train)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)

    # 3. Generate random, shuffled indices
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()

    # 4. Create the subsets using the correct parent dataset!
    train_dataset = Subset(dataset_train, indices[:train_size])
    val_dataset = Subset(dataset_eval, indices[train_size : train_size + val_size])
    test_dataset = Subset(dataset_eval, indices[train_size + val_size :])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset_dir = get_dataset_dir()

    preprocessor = ExpWPreprocessor(
        raw_metadata_path=os.path.join(dataset_dir, "label.lst"),
        stratified_csv_path=os.path.join(
            dataset_dir,
            "Stratified_10k_Metadata.csv",
        ),
        input_dir_pattern=os.path.join(dataset_dir, "origin", "*.jpg"),
        output_dir=os.path.join(
            dataset_dir,
            "Stratified_10k_Cleaned_64x64",
        ),
    )

    print("Building balanced metadata...")
    images = preprocessor.build_balanced_metadata(samples_per_class=1500)

    print("Finding input directory...")
    input_dir = preprocessor.find_input_dir()

    print("Running preprocessing...")
    preprocessor.run(images, input_dir)

    print("Testing dataloader...")

    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("images:", images.shape)
    print("labels:", labels.shape)
