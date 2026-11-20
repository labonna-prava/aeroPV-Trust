from setuptools import setup, find_packages

setup(
    name="aeropv-trust",
    version="1.0.0",
    description="Multimodal UAV Photovoltaic Anomaly Inspection & Trustworthy XAI Framework",
    author="Labonna Prava",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "opencv-python>=4.8.0",
        "tifffile>=2023.7.0",
        "scikit-image>=0.21.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.30.0",
        "tqdm>=4.65.0",
    ],
)
