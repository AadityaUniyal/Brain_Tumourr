from setuptools import setup, find_packages

setup(
    name="neurovision_elite",
    version="2.0.0",
    description="State-of-the-Art Dual-Task Brain Tumor Segmentation, Classification, and Clinical XAI System",
    author="Aaditya Uniyal",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pillow>=10.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
)
