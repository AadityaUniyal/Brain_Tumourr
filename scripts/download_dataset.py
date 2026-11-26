"""
Automated downloader and extractor for Jun Cheng Figshare Brain Tumor Dataset (3,064 MRI scans).
"""
import os
import shutil
import zipfile
import requests
import time

DATASET_FILES = [
    {
        "name": "brainTumorDataPublic_1-766.zip",
        "url": "https://ndownloader.figshare.com/files/3381290",
        "expected_mb": 204.5,
    },
    {
        "name": "brainTumorDataPublic_767-1532.zip",
        "url": "https://ndownloader.figshare.com/files/3381296",
        "expected_mb": 207.8,
    },
    {
        "name": "brainTumorDataPublic_1533-2298.zip",
        "url": "https://ndownloader.figshare.com/files/3381293",
        "expected_mb": 205.6,
    },
    {
        "name": "brainTumorDataPublic_2299-3064.zip",
        "url": "https://ndownloader.figshare.com/files/3381302",
        "expected_mb": 220.9,
    },
    {
        "name": "cvind.mat",
        "url": "https://ndownloader.figshare.com/files/7005344",
        "expected_mb": 0.01,
    },
]


def download_file(url, target_path, expected_mb):
    print(f"Downloading {os.path.basename(target_path)} (~{expected_mb:.1f} MB)...")
    start_time = time.time()
    # Note: Do not pass custom auth/user-agent headers because Cloudflare/AWS S3 presigned URLs reject modified signatures
    with requests.get(url, stream=True, allow_redirects=True, timeout=30) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

    elapsed = time.time() - start_time
    mb_downloaded = downloaded / (1024 * 1024)
    speed = mb_downloaded / max(elapsed, 0.1)
    print(f" -> Completed {mb_downloaded:.1f} MB in {elapsed:.1f}s ({speed:.2f} MB/s)")


def main():
    raw_dir = os.path.join("data", "raw")
    zips_dir = os.path.join(raw_dir, "zips")
    mat_dir = os.path.join(raw_dir, "mat_files")

    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)

    print("=== Downloading 3,064 Brain Tumor MRI Scans from Figshare ===")
    for item in DATASET_FILES:
        is_zip = item["name"].endswith(".zip")
        dest_dir = zips_dir if is_zip else raw_dir
        target_path = os.path.join(dest_dir, item["name"])

        # Check if already downloaded and valid
        if os.path.exists(target_path):
            file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
            if file_size_mb >= item["expected_mb"] * 0.95:
                print(f"[SKIP] {item['name']} already exists ({file_size_mb:.1f} MB).")
                continue
            else:
                print(f"[REPLACE] Corrupt/partial file found for {item['name']}. Re-downloading...")
                os.remove(target_path)

        download_file(item["url"], target_path, item["expected_mb"])

    # Extract all zip files into mat_files
    print("\n=== Extracting Archives to data/raw/mat_files/ ===")
    for item in DATASET_FILES:
        if item["name"].endswith(".zip"):
            zip_path = os.path.join(zips_dir, item["name"])
            print(f"Extracting {item['name']}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(mat_dir)

    total_mat_files = len([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
    print(f"\n✅ All Archives Extracted! Total .mat MRI scans in {mat_dir}: {total_mat_files}")

    # Immediately trigger EDA analysis
    print("\n=== Launching Automated EDA Pipeline ===")
    from scripts.eda_analysis import run_eda
    run_eda(mat_dir=mat_dir)


if __name__ == "__main__":
    main()
