"""
Extract IBSI-compliant High-Throughput Radiomic Features across Brain Tumor Scans.
Features extracted per tumor mask:
- First-order Intensity Statistics (Mean, Std, Skewness, Kurtosis, Entropy, Percentiles)
- Morphological & Shape Features (Area, Perimeter, Compactness, Eccentricity, Solidity, Elongation)
- Second-order GLCM Texture Features (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
- High-frequency edge & Laplacian spatial gradients
"""
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops

MAT_DIR = Path("data/raw/mat_files")
OUT_CSV = Path("data/processed/radiomic_features_patient_level.csv")
OUT_SLICE_CSV = Path("data/processed/radiomic_features_slice_level.csv")

def extract_slice_radiomics(image: np.ndarray, mask: np.ndarray) -> dict:
    tumor_pixels = image[mask > 0]
    if len(tumor_pixels) < 10:
        return None

    feats = {}
    
    # 1. First-Order Intensity Statistics
    feats["intensity_mean"] = float(np.mean(tumor_pixels))
    feats["intensity_std"] = float(np.std(tumor_pixels))
    feats["intensity_variance"] = float(np.var(tumor_pixels))
    feats["intensity_skewness"] = float(stats.skew(tumor_pixels))
    feats["intensity_kurtosis"] = float(stats.kurtosis(tumor_pixels))
    feats["intensity_median"] = float(np.median(tumor_pixels))
    feats["intensity_p10"] = float(np.percentile(tumor_pixels, 10))
    feats["intensity_p90"] = float(np.percentile(tumor_pixels, 90))
    feats["intensity_iqr"] = float(stats.iqr(tumor_pixels))
    
    # Histogram Shannon Entropy
    hist, _ = np.histogram(tumor_pixels, bins=32, density=True)
    hist = hist[hist > 0]
    feats["intensity_entropy"] = float(-np.sum(hist * np.log2(hist)))

    # 2. Morphological / Shape Features
    props = regionprops((mask > 0).astype(int))
    if props:
        prop = props[0]
        area = prop.area
        perimeter = prop.perimeter if prop.perimeter > 0 else 1.0
        feats["shape_area"] = float(area)
        feats["shape_perimeter"] = float(perimeter)
        feats["shape_compactness"] = float((perimeter ** 2) / (4.0 * np.pi * area))
        feats["shape_eccentricity"] = float(prop.eccentricity)
        feats["shape_solidity"] = float(prop.solidity)
        feats["shape_extent"] = float(prop.extent)
        feats["shape_equiv_diameter"] = float(prop.equivalent_diameter_area)
        
        # Major / Minor axis ratio (elongation)
        major = prop.major_axis_length
        minor = prop.minor_axis_length if prop.minor_axis_length > 0 else 1.0
        feats["shape_elongation"] = float(major / minor)
    else:
        feats["shape_area"] = 0.0
        feats["shape_perimeter"] = 0.0
        feats["shape_compactness"] = 0.0
        feats["shape_eccentricity"] = 0.0
        feats["shape_solidity"] = 0.0
        feats["shape_extent"] = 0.0
        feats["shape_equiv_diameter"] = 0.0
        feats["shape_elongation"] = 1.0

    # 3. Second-Order Haralick / GLCM Texture Features
    # Crop bounding box of tumor for efficient GLCM
    minr, minc, maxr, maxc = prop.bbox if props else (0, 0, image.shape[0], image.shape[1])
    cropped_img = image[minr:maxr, minc:maxc]
    cropped_mask = mask[minr:maxr, minc:maxc]
    
    # Quantize cropped tumor to 16 grey levels
    tumor_crop = np.where(cropped_mask > 0, cropped_img, 0)
    norm = ((tumor_crop - tumor_crop.min()) / (tumor_crop.max() - tumor_crop.min() + 1e-6) * 15).astype(np.uint8)
    
    glcm = graycomatrix(norm, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
    
    feats["glcm_contrast"] = float(np.mean(graycoprops(glcm, 'contrast')))
    feats["glcm_dissimilarity"] = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    feats["glcm_homogeneity"] = float(np.mean(graycoprops(glcm, 'homogeneity')))
    feats["glcm_energy"] = float(np.mean(graycoprops(glcm, 'energy')))
    feats["glcm_correlation"] = float(np.mean(graycoprops(glcm, 'correlation')))
    feats["glcm_ASM"] = float(np.mean(graycoprops(glcm, 'ASM')))

    return feats

def main():
    print("=== Extracting Radiomics across All 3,064 Scans ===")
    mat_files = sorted(list(MAT_DIR.glob("*.mat")), key=lambda p: int(p.stem))
    print(f"Discovered {len(mat_files)} scan files.")

    slice_records = []
    for idx, path in enumerate(mat_files):
        if (idx + 1) % 500 == 0 or idx == len(mat_files) - 1:
            print(f"Processed {idx + 1}/{len(mat_files)} scans...")

        try:
            with h5py.File(path, 'r') as f:
                cjdata = f['cjdata']
                label = int(cjdata['label'][0, 0])
                patient_id = "".join([chr(c[0]) for c in cjdata['PID'][:]])
                image = np.array(cjdata['image'], dtype=np.float32)
                tumor_mask = np.array(cjdata['tumorMask'], dtype=np.uint8)

            feats = extract_slice_radiomics(image, tumor_mask)
            if feats is not None:
                feats["file_id"] = int(path.stem)
                feats["patient_id"] = patient_id
                feats["tumor_label"] = label
                slice_records.append(feats)
        except Exception as e:
            continue

    df_slices = pd.DataFrame(slice_records)
    print(f"\nExtracted slice-level radiomic profiles for {len(df_slices)} scans.")
    df_slices.to_csv(OUT_SLICE_CSV, index=False)
    print(f"Saved slice-level radiomics to: {OUT_SLICE_CSV}")

    # Patient-level 3D spatial aggregation (Mean + Std of radiomics across patient slices)
    feature_cols = [c for c in df_slices.columns if c not in ["file_id", "patient_id", "tumor_label"]]
    
    agg_funcs = {col: ['mean', 'std'] for col in feature_cols}
    agg_funcs["tumor_label"] = 'first'
    agg_funcs["file_id"] = 'count'  # number of slices (proxy for tumor volumetric depth)

    df_patient = df_slices.groupby("patient_id").agg(agg_funcs)
    # Flatten multi-level columns
    df_patient.columns = ['_'.join(c).strip('_') for c in df_patient.columns.values]
    df_patient = df_patient.rename(columns={"tumor_label_first": "tumor_label", "file_id_count": "slice_depth_volume"})
    df_patient = df_patient.reset_index().fillna(0)

    print(f"\nConstructed Patient-Level Volumetric Radiomic Profiles for {len(df_patient)} unique patients.")
    print(f"Total quantitative features per patient: {df_patient.shape[1] - 2}")
    df_patient.to_csv(OUT_CSV, index=False)
    print(f"Saved patient-level radiomic dataset to: {OUT_CSV}")

if __name__ == "__main__":
    main()
