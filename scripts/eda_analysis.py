"""
Comprehensive Exploratory Data Analysis (EDA) for Jun Cheng Figshare Dataset.
Generates:
1. data/processed/dataset_metadata.csv (Complete audit of all 3,064 scans)
2. reports/figures/eda/class_distribution.png
3. reports/figures/eda/slices_per_patient_distribution.png
4. reports/figures/eda/tumor_size_boxplot.png
5. reports/figures/eda/sample_mri_gallery.png
"""
import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

CLASS_MAP = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
PIXEL_SPACING_MM = 0.49  # 0.49 mm x 0.49 mm per pixel
PIXEL_AREA_MM2 = PIXEL_SPACING_MM * PIXEL_SPACING_MM


def parse_mat_file(file_path):
    """Robustly extracts all fields from .mat file."""
    try:
        with h5py.File(file_path, "r") as f:
            cjdata = f["cjdata"]
            image = np.array(cjdata["image"], dtype=np.float32)
            mask = np.array(cjdata["tumorMask"], dtype=np.uint8)
            label = int(np.array(cjdata["label"])[0][0])
            pid_raw = cjdata.get("PID", None)
            if pid_raw is not None:
                pid = "".join(chr(c[0]) for c in np.array(pid_raw))
            else:
                pid = os.path.basename(file_path).split(".")[0]
            return image, mask, label, pid
    except Exception:
        if loadmat is not None:
            mat = loadmat(file_path)
            cjdata = mat["cjdata"]
            image = cjdata["image"][0, 0].astype(np.float32)
            mask = cjdata["tumorMask"][0, 0].astype(np.uint8)
            label = int(cjdata["label"][0, 0][0, 0])
            pid_raw = cjdata["PID"][0, 0]
            pid = str(pid_raw[0]) if len(pid_raw) > 0 else os.path.basename(file_path).split(".")[0]
            return image, mask, label, pid
        raise


def run_eda(mat_dir="data/raw/mat_files", output_dir="reports/figures/eda", csv_out="data/processed/dataset_metadata.csv"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    print(f"Found {len(mat_files)} .mat files. Parsing metadata...")

    records = []
    sample_images = {1: [], 2: [], 3: []}

    for fpath in tqdm(mat_files, desc="Auditing Scans"):
        fname = os.path.basename(fpath)
        image, mask, label, pid = parse_mat_file(fpath)

        tumor_pixels = int(np.sum(mask > 0))
        tumor_area_mm2 = round(tumor_pixels * PIXEL_AREA_MM2, 2)

        record = {
            "file_name": fname,
            "file_path": fpath,
            "patient_id": pid,
            "label_id": label,
            "class_name": CLASS_MAP.get(label, "Unknown"),
            "image_height": image.shape[0],
            "image_width": image.shape[1],
            "min_intensity": float(np.min(image)),
            "max_intensity": float(np.max(image)),
            "mean_intensity": round(float(np.mean(image)), 2),
            "std_intensity": round(float(np.std(image)), 2),
            "tumor_pixel_count": tumor_pixels,
            "tumor_area_mm2": tumor_area_mm2,
        }
        records.append(record)

        # Keep 3 samples per class for visual gallery
        if len(sample_images[label]) < 3:
            sample_images[label].append((image, mask, fname, pid))

    df = pd.DataFrame(records)
    df.to_csv(csv_out, index=False)
    print(f"\n✅ Metadata saved to {csv_out} ({len(df)} rows)")

    # Print Clinical Statistics
    print("\n" + "=" * 50)
    print("🧠 CLINICAL DATASET SUMMARY")
    print("=" * 50)
    print(f"Total MRI Slices: {len(df)}")
    print(f"Total Unique Patients: {df['patient_id'].nunique()}")

    print("\nSlice Count per Class:")
    print(df["class_name"].value_counts())

    print("\nUnique Patients per Class:")
    patient_class_counts = df.groupby("class_name")["patient_id"].nunique()
    print(patient_class_counts)

    print("\nTumor Area Statistics (mm²):")
    print(df.groupby("class_name")["tumor_area_mm2"].describe()[["count", "mean", "std", "min", "50%", "max"]])

    # 1. Class Distribution Plot (Slices vs Patients)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    slice_counts = df["class_name"].value_counts()
    sns.barplot(x=slice_counts.index, y=slice_counts.values, ax=axes[0], palette="crest")
    axes[0].set_title("Distribution of MRI Slices per Class", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Slice Count")
    for i, v in enumerate(slice_counts.values):
        axes[0].text(i, v + 25, str(v), ha="center", fontweight="bold")

    sns.barplot(x=patient_class_counts.index, y=patient_class_counts.values, ax=axes[1], palette="viridis")
    axes[1].set_title("Distribution of Unique Patients per Class", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Patient Count")
    for i, v in enumerate(patient_class_counts.values):
        axes[1].text(i, v + 2, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # 2. Slices per Patient Distribution
    plt.figure(figsize=(10, 5))
    slices_per_patient = df.groupby("patient_id").size()
    sns.histplot(slices_per_patient, bins=25, kde=True, color="#2b5c8f")
    plt.title("Distribution of MRI Slices per Patient (Why Patient-Level Splitting is Vital)", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Slices from Same Patient")
    plt.ylabel("Patient Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slices_per_patient_distribution.png"), dpi=300)
    plt.close()

    # 3. Tumor Area Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="class_name", y="tumor_area_mm2", palette="Set2")
    plt.title("Tumor Cross-Sectional Area (mm²) by Pathological Subtype", fontsize=13, fontweight="bold")
    plt.ylabel("Tumor Area (mm²)")
    plt.xlabel("Tumor Classification")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tumor_size_boxplot.png"), dpi=300)
    plt.close()

    # 4. Sample MRI Gallery with Ground Truth Masks
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for row, label_id in enumerate([1, 2, 3]):
        cname = CLASS_MAP[label_id]
        samples = sample_images[label_id]
        for col in range(3):
            ax = axes[row, col]
            if col < len(samples):
                img, mask, fname, pid = samples[col]
                # Normalize image for display
                disp_img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
                ax.imshow(disp_img, cmap="bone")
                # Contour overlay
                ax.contour(mask, colors="red", linewidths=1.5)
                ax.set_title(f"{cname} | PID: {pid}\n{fname}", fontsize=9, fontweight="bold")
            ax.axis("off")

    plt.suptitle("Sample T1-CE MRI Scans with Radiologist Ground-Truth Contours (Red)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_mri_gallery.png"), dpi=300)
    plt.close()

    print(f"\n📊 All 4 publication-grade EDA figures exported to {output_dir}/")


if __name__ == "__main__":
    run_eda()
