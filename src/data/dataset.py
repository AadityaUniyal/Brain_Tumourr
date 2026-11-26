"""
PyTorch Dataset and Medical Augmentation Pipeline.
Supports dual-task learning: classification label + binary segmentation mask.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

# Label mapping for Figshare dataset:
# 1 -> Meningioma (index 0)
# 2 -> Glioma (index 1)
# 3 -> Pituitary (index 2)
CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]
LABEL_MAP = {1: 0, 2: 1, 3: 2}


class BrainTumorDataset(Dataset):
    """
    Dual-Task Medical Dataset:
    Returns:
        image: Tensor of shape [3, H, W]
        mask: Tensor of shape [1, H, W] (segmentation ground truth)
        label: Tensor of shape [] (classification class 0, 1, or 2)
        patient_id: str
    """

    def __init__(self, items, transforms=None, target_size=(256, 256)):
        """
        items: List of dicts, each containing:
            'image': np.ndarray or file_path
            'mask': np.ndarray or None
            'label': int (0, 1, or 2)
            'patient_id': str
        """
        self.items = items
        self.transforms = transforms
        self.target_size = target_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = item["image"]
        mask = item.get("mask", None)
        label = item["label"]
        patient_id = item.get("patient_id", f"P_{idx}")

        # Ensure image is resized and [H, W, 1] or [H, W, 3]
        if isinstance(image, str):
            # Load from path if stored as path
            from src.data.preprocessor import MedicalImagePreprocessor
            prep = MedicalImagePreprocessor(target_size=self.target_size)
            image = prep.preprocess(image, filename=image)

        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            if mask.shape[:2] != self.target_size:
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(self.target_size, dtype=np.float32)

        # Expand image to 3 channels for backbone architectures
        if len(image.shape) == 2:
            image_3c = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            image_3c = np.repeat(image, 3, axis=-1)
        else:
            image_3c = image

        # Apply Albumentations augmentations if provided
        if self.transforms is not None:
            augmented = self.transforms(image=image_3c, mask=mask)
            image_3c = augmented["image"]
            mask = augmented["mask"]

        # Convert to PyTorch tensors
        # Albumentations with ToTensorV2 converts to [C, H, W]
        if not isinstance(image_3c, torch.Tensor):
            image_tensor = torch.from_numpy(image_3c).permute(2, 0, 1).float()
        else:
            image_tensor = image_3c.float()

        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask_tensor = mask.unsqueeze(0).float() if mask.ndim == 2 else mask.float()

        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "patient_id": patient_id,
        }


def get_medical_transforms(is_train=True):
    """
    Returns robust medical augmentations specifically suitable for brain MRI:
    - Random rotation (+/- 15 deg)
    - Horizontal flip (axial brain scans are anatomically symmetric)
    - Random brightness & contrast (simulates scanner variations across GE/Siemens)
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if is_train:
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
    except ImportError:
        # Fallback if albumentations is not installed
        return None
