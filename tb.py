import copy
import os
import random
import shutil
import zipfile
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import logging
import yaml
import optuna
from optuna.trial import Trial

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import numpy as np


class DoubleConv(nn.Module):
    """
        A module that performs two consecutive convolution operations followed by ReLU activations.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Methods:
            forward(x: torch.Tensor) -> torch.Tensor:
                Forward pass through the double convolutional layers.
        """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_op(x)


class DownSample(nn.Module):
    """
        A module that performs downsampling using a double convolution followed by max pooling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Methods:
            forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                Forward pass through the downsampling layers.
        """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    """
        A module that performs upsampling using a transposed convolution followed by a double convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Methods:
            forward(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                Forward pass through the upsampling layers.
        """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle cases where input dimensions don't match exactly
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    """
        A U-Net model for image segmentation.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate for regularization.

        Methods:
            forward(x: torch.Tensor) -> torch.Tensor:
                Forward pass through the U-Net model.
        """

    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.dropout = nn.Dropout(dropout_rate)
        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)
        b = self.dropout(b)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        return self.out(up_4)


class DRIVEDataset(Dataset):
    """
        A custom dataset for loading images and masks from the DRIVE dataset.

        Args:
            root_path (str): Root directory of the dataset.
            limit (Optional[int]): Limit the number of samples to load.

        Methods:
            __getitem__(index: int) -> Tuple[torch.Tensor, torch.Tensor]:
                Get a sample from the dataset.
            __len__() -> int:
                Get the number of samples in the dataset.
        """

    def __init__(self, root_path=os.getcwd(), limit: Optional[int] = None):
        self.root_path = root_path
        self.limit = limit

        # Check if the paths exist
        train_images_path = os.path.join(root_path, "data\\train\\images")
        train_masks_path = os.path.join(root_path, "data\\train\\mask")

        if not os.path.exists(train_images_path) or not os.path.exists(train_masks_path):
            raise FileNotFoundError(f"Dataset directories not found at {root_path}")

        self.images = sorted([
            os.path.join(train_images_path, i)
            for i in os.listdir(train_images_path)
            if i.endswith(('.jpg', '.png', '.jpeg', '.tif'))
        ])[:self.limit]

        self.masks = sorted([
            os.path.join(train_masks_path, i)
            for i in os.listdir(train_masks_path)
            if i.endswith(('.jpg', '.png', '.jpeg', '.gif'))
        ])[:self.limit]

        if len(self.images) != len(self.masks):
            print(len(self.images), len(self.masks))
            raise ValueError("Number of images and masks do not match")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            img = Image.open(self.images[index]).convert("RGB")
            mask = Image.open(self.masks[index]).convert("L")

            if img is None or mask is None:
                raise ValueError(f"Failed to load image or mask at index {index}")

            return self.transform(img), self.transform(mask)
        except Exception as e:
            logging.error(f"Error loading data at index {index}: {e}")
            raise e

    def __len__(self) -> int:
        return min(len(self.images), self.limit)



def DICE(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-07) -> torch.Tensor:
    """
        Calculate the DICE coefficient between the prediction and target tensors.

        Args:
            prediction (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Ground truth tensor.
            epsilon (float): Small value to avoid division by zero.

        Returns:
            torch.Tensor: DICE coefficient.
        """
    prediction = torch.sigmoid(prediction)  # Apply sigmoid for binary segmentation
    prediction = (prediction > 0.5).float()

    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)

    return (2. * intersection + epsilon) / (union + epsilon)


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: str,
                epochs: int = 10,
                patience: int = 5) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
        Train the model and validate it after each epoch.

        Args:
            model (nn.Module): The model to train.
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function.
            device (str): Device to run the training on ('cpu' or 'cuda').
            epochs (int): Number of epochs to train.
            patience (int): Number of epochs to wait for improvement before early stopping.

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]:
                Training losses, training DICE scores, validation losses, validation DICE scores.
        """
    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_running_loss = 0
        train_running_dc = 0

        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Training]')
        for idx, (img, mask) in enumerate(train_pbar):
            try:
                img = img.float().to(device)
                mask = mask.float().to(device)

                y_pred = model(img)
                optimizer.zero_grad()

                dc = DICE(y_pred, mask)
                loss = criterion(y_pred, mask)

                loss.backward()
                optimizer.step()

                train_running_loss += loss.item()
                train_running_dc += dc.item()

                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dc.item():.4f}'
                })

            except RuntimeError as e:
                logging.error(f"Runtime error during training: {e}")
                continue

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        # Validation phase
        model.eval()
        val_running_loss = 0
        val_running_dc = 0

        val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]')
        with torch.no_grad():
            for idx, (img, mask) in enumerate(val_pbar):
                try:
                    img = img.float().to(device)
                    mask = mask.float().to(device)

                    y_pred = model(img)
                    loss = criterion(y_pred, mask)
                    dc = DICE(y_pred, mask)

                    val_running_loss += loss.item()
                    val_running_dc += dc.item()

                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'dice': f'{dc.item():.4f}'
                    })

                except RuntimeError as e:
                    logging.error(f"Runtime error during validation: {e}")
                    continue

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            model.load_state_dict(best_model_state)
            break

        print("-" * 30)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training DICE: {train_dc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation DICE: {val_dc:.4f}")
        print("-" * 30)

    return train_losses, train_dcs, val_losses, val_dcs


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

    # Configuration
    root_path = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Dataset setup
        train_ds = DRIVEDataset(root_path, limit=None)
        gen = torch.Generator().manual_seed(25)

        # Calculate lengths for splits
        total_length = len(train_ds)
        train_length = int(0.8 * total_length)
        val_test_length = total_length - train_length
        val_length = val_test_length // 2
        test_length = val_test_length - val_length

        # Perform splits
        train_ds, val_test_ds = random_split(train_ds, [train_length, val_test_length], generator=gen)
        val_ds, test_ds = random_split(val_test_ds, [val_length, test_length], generator=gen)

        train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=8, shuffle=False)
        test_dataloader = DataLoader(test_ds, batch_size=8, shuffle=False)

        # Model setup
        model = UNet(in_channels=3, num_classes=1, dropout_rate=0.1).to(device)

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        train_losses, train_dcs, val_losses, val_dcs = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=10,
            patience=5
        )

        # Save the model
        model_path = os.path.join(root_path, "unet_model.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

        # Evaluate on the test set
        model.eval()
        test_running_loss = 0
        test_running_dc = 0

        with torch.no_grad():
            for idx, (img, mask) in enumerate(tqdm(test_dataloader, desc="Testing")):
                img = img.float().to(device)
                mask = mask.float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                dc = DICE(y_pred, mask)

                test_running_loss += loss.item()
                test_running_dc += dc.item()

        test_loss = test_running_loss / (idx + 1)
        test_dc = test_running_dc / (idx + 1)

        logging.info(f"Test Loss: {test_loss:.4f}")
        logging.info(f"Test DICE: {test_dc:.4f}")

        # Plot training and validation metrics
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(train_dcs, label='Train DICE')
        plt.plot(val_dcs, label='Validation DICE')
        plt.legend()
        plt.title('DICE Score over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('DICE Score')
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()