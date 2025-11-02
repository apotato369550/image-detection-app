"""
Setup configuration for the Computer Vision Application.

This file provides package metadata and installation configuration
for the computer vision application.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="computer-vision-app",
    version="1.0.0",
    author="Computer Vision Developer",
    author_email="developer@example.com",
    description="A complete computer vision application for real-time object detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/computer-vision-app",
    project_urls={
        "Documentation": "https://github.com/yourusername/computer-vision-app#readme",
        "Source": "https://github.com/yourusername/computer-vision-app",
        "Tracker": "https://github.com/yourusername/computer-vision-app/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cv-app=src.main:main",
            "asl-app=src.asl_main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json"],
    },
    keywords=[
        "computer-vision",
        "object-detection",
        "tensorflow",
        "tensorflow-lite",
        "opencv",
        "real-time",
        "machine-learning",
        "deep-learning",
        "image-processing",
    ],
    zip_safe=False,
)