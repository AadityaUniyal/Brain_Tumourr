"""
Medical Image Preprocessor:
Robustly parses and standardizes multiple medical imaging modalities:
- DICOM (.dcm) files with RescaleSlope and RescaleIntercept
- NIfTI (.nii, .nii.gz) volumetric neuroimaging files
- Jun Cheng Figshare (.mat) MATLAB files
- Standard radiological exports (.png, .jpg, .jpeg)
"""
import io
import os
import tempfile
import numpy as np
import cv2
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


class MedicalImagePreprocessor:
    """Standardizes input medical images into calibrated 2D intensity matrices."""

    def __init__(self, target_size=(256, 256), normalize_method="minmax"):
        self.target_size = target_size
        self.normalize_method = normalize_method

    def load_dicom(self, file_source):
        """Loads a DICOM file from file path or bytes buffer."""
        if pydicom is None:
            raise ImportError("pydicom is required to parse .dcm files.")
        if isinstance(file_source, bytes):
            ds = pydicom.dcmread(io.BytesIO(file_source))
        else:
            ds = pydicom.dcmread(file_source)

        pixel_array = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        pixel_array = pixel_array * float(slope) + float(intercept)
        return pixel_array

    def load_nifti(self, file_source, slice_idx=None):
        """Loads a NIfTI volume from path or bytes and extracts an axial slice."""
        if nib is None:
            raise ImportError("nibabel is required to parse .nii / .nii.gz files.")

        if isinstance(file_source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(file_source)
                tmp_path = tmp.name
            try:
                nii = nib.load(tmp_path)
                data = nii.get_fdata().astype(np.float32)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            nii = nib.load(file_source)
            data = nii.get_fdata().astype(np.float32)

        # Axial slice extraction
        if slice_idx is None:
            slice_idx = data.shape[2] // 2
        slice_idx = max(0, min(slice_idx, data.shape[2] - 1))
        return data[:, :, slice_idx]

    def load_mat(self, file_source):
        """
        Parses Jun Cheng Figshare .mat format.
        Returns:
            image (np.ndarray), mask (np.ndarray), label (int), patient_id (str)
        """
        if isinstance(file_source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
                tmp.write(file_source)
                tmp_path = tmp.name
            try:
                return self._parse_mat_file(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            return self._parse_mat_file(file_source)

    def _parse_mat_file(self, file_path):
        """Internal parser supporting both HDF5 (v7.3) and legacy MAT formats."""
        # Try h5py first (MATLAB v7.3)
        if h5py is not None:
            try:
                with h5py.File(file_path, "r") as f:
                    cjdata = f["cjdata"]
                    image = np.array(cjdata["image"], dtype=np.float32)
                    mask = np.array(cjdata["tumorMask"], dtype=np.uint8)
                    label = int(np.array(cjdata["label"])[0][0])
                    pid_data = cjdata.get("PID", None)
                    if pid_data is not None:
                        pid = "".join(chr(c[0]) for c in np.array(pid_data))
                    else:
                        pid = os.path.basename(file_path).split(".")[0]
                    return image, mask, label, pid
            except Exception:
                pass

        # Fallback to scipy.io.loadmat
        if loadmat is not None:
            mat = loadmat(file_path)
            cjdata = mat["cjdata"]
            image = cjdata["image"][0, 0].astype(np.float32)
            mask = cjdata["tumorMask"][0, 0].astype(np.uint8)
            label = int(cjdata["label"][0, 0][0, 0])
            pid_raw = cjdata["PID"][0, 0]
            pid = str(pid_raw[0]) if len(pid_raw) > 0 else os.path.basename(file_path).split(".")[0]
            return image, mask, label, pid

        raise RuntimeError("Neither h5py nor scipy.io.loadmat could parse the .mat file.")

    def load_standard_image(self, file_source):
        """Loads PNG, JPG, or BMP into grayscale float array."""
        if isinstance(file_source, bytes):
            img = Image.open(io.BytesIO(file_source)).convert("L")
        else:
            img = Image.open(file_source).convert("L")
        return np.array(img, dtype=np.float32)

    def normalize(self, image):
        """Applies robust min-max or z-score normalization."""
        if self.normalize_method == "minmax":
            p_min, p_max = np.min(image), np.max(image)
            if p_max > p_min:
                return (image - p_min) / (p_max - p_min)
            return np.zeros_like(image)
        elif self.normalize_method == "zscore":
            mean, std = np.mean(image), np.std(image)
            if std > 1e-6:
                return (image - mean) / std
            return np.zeros_like(image)
        return image

    def preprocess(self, file_source, filename="scan.jpg"):
        """
        Unified preprocessing pipeline:
        Reads file -> Extracts 2D array -> Normalizes -> Resizes to target size.
        Returns:
            preprocessed_image: float32 ndarray of shape (H, W), values in [0, 1]
        """
        ext = filename.lower().split(".")[-1]
        if ext == "dcm":
            raw = self.load_dicom(file_source)
        elif ext in ["nii", "gz"]:
            raw = self.load_nifti(file_source)
        elif ext == "mat":
            raw, _, _, _ = self.load_mat(file_source)
        else:
            raw = self.load_standard_image(file_source)

        normalized = self.normalize(raw)
        resized = cv2.resize(normalized, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.float32)
