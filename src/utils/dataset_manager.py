"""
Dataset Manager Module

Handles downloading and caching of computer vision datasets, particularly COCO annotations.
Provides structured access to dataset labels and metadata for object detection tasks.
"""

import json
import logging
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Category:
    """Represents a COCO dataset category."""
    id: int
    name: str
    supercategory: str


@dataclass
class Annotation:
    """Represents a COCO dataset annotation."""
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height]
    area: float
    iscrowd: int


@dataclass
class ImageInfo:
    """Represents image information from COCO dataset."""
    id: int
    width: int
    height: int
    file_name: str
    license: Optional[int] = None
    coco_url: Optional[str] = None


class DatasetManager:
    """
    Manages downloading and caching of computer vision datasets.

    Currently supports COCO dataset annotations with efficient caching
    and structured data access.
    """

    def __init__(self, data_dir: str = "data") -> None:
        """
        Initialize the DatasetManager.

        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.session = requests.Session()

        # COCO dataset configurations
        self.dataset_configs = {
            "coco_2017_train": {
                "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "filename": "annotations_trainval2017.zip",
                "description": "COCO 2017 Train/Val annotations",
                "expected_files": [
                    "annotations/instances_train2017.json",
                    "annotations/instances_val2017.json",
                    "annotations/person_keypoints_train2017.json",
                    "annotations/person_keypoints_val2017.json",
                    "annotations/captions_train2017.json",
                    "annotations/captions_val2017.json"
                ]
            },
            "coco_2017_val": {
                "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "filename": "annotations_trainval2017.zip",
                "description": "COCO 2017 Train/Val annotations (same as train)",
                "expected_files": [
                    "annotations/instances_val2017.json"
                ]
            },
            "coco_2014_train": {
                "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                "filename": "annotations_trainval2014.zip",
                "description": "COCO 2014 Train/Val annotations",
                "expected_files": [
                    "annotations/instances_train2014.json",
                    "annotations/instances_val2014.json"
                ]
            }
        }

    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset already exists locally.

        Args:
            dataset_name: Name of the dataset to check

        Returns:
            bool: True if dataset exists, False otherwise
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False

        config = self.dataset_configs[dataset_name]

        # Check if expected files exist
        for expected_file in config["expected_files"]:
            file_path = self.data_dir / expected_file
            if not file_path.exists():
                logger.info(f"Dataset {dataset_name} missing file: {expected_file}")
                return False

        logger.info(f"Dataset {dataset_name} exists locally")
        return True

    def download_file(self, url: str, filepath: Path) -> bool:
        """
        Download a file with progress tracking.

        Args:
            url: URL to download from
            filepath: Local path to save the file

        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            logger.info(f"Downloading from: {url}")
            logger.info(f"Saving to: {filepath}")

            # Get file size for progress tracking
            head_response = self.session.head(url)
            total_size = int(head_response.headers.get('content-length', 0))

            # Download with progress tracking
            with self.session.get(url, stream=True) as response:
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"Download progress: {progress:.1f}%")

            logger.info(f"Download completed: {filepath}")
            return True

        except requests.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {str(e)}")
            return False

    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """
        Extract a ZIP archive.

        Args:
            zip_path: Path to the ZIP file
            extract_to: Directory to extract to

        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            logger.info(f"Extracting {zip_path} to {extract_to}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total files for progress tracking
                file_list = zip_ref.namelist()
                total_files = len(file_list)

                for i, file in enumerate(file_list):
                    zip_ref.extract(file, extract_to)
                    if i % 10 == 0:  # Log every 10 files
                        progress = ((i + 1) / total_files) * 100
                        logger.info(f"Extraction progress: {progress:.1f}%")

            logger.info("Extraction completed successfully")
            return True

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return False

    def download_dataset(self, dataset_name: str) -> Optional[str]:
        """
        Download a dataset if it doesn't exist locally.

        Args:
            dataset_name: Name of the dataset to download

        Returns:
            Optional[str]: Path to the dataset directory if successful, None otherwise
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.error(f"Available datasets: {list(self.dataset_configs.keys())}")
            return None

        # Check if dataset already exists
        if self.dataset_exists(dataset_name):
            logger.info(f"Dataset {dataset_name} already exists")
            return str(self.data_dir)

        logger.info(f"Downloading dataset: {dataset_name}")

        config = self.dataset_configs[dataset_name]
        url = config["url"]
        filename = config["filename"]

        # Download the dataset archive
        archive_path = self.data_dir / filename

        if not self.download_file(url, archive_path):
            return None

        # Extract the archive
        if not self.extract_zip(archive_path, self.data_dir):
            return None

        # Clean up archive file
        try:
            archive_path.unlink()
            logger.info(f"Cleaned up archive: {archive_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up archive: {str(e)}")

        # Verify dataset exists
        if not self.dataset_exists(dataset_name):
            logger.error(f"Dataset files not found after extraction")
            return None

        logger.info(f"Dataset {dataset_name} downloaded successfully")
        return str(self.data_dir)

    def load_coco_annotations(self, annotation_file: str) -> Dict[str, Any]:
        """
        Load and parse COCO annotations from a JSON file.

        Args:
            annotation_file: Path to the annotation JSON file

        Returns:
            Dict[str, Any]: Parsed annotation data
        """
        try:
            file_path = self.data_dir / annotation_file
            if not file_path.exists():
                logger.error(f"Annotation file not found: {file_path}")
                return {}

            logger.info(f"Loading annotations from: {file_path}")

            with open(file_path, 'r') as f:
                data = json.load(f)

            logger.info(f"Loaded {len(data.get('annotations', []))} annotations")
            logger.info(f"Found {len(data.get('categories', []))} categories")
            logger.info(f"Found {len(data.get('images', []))} images")

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            return {}

    def get_categories(self, annotation_file: str) -> List[Category]:
        """
        Get all categories from COCO annotations.

        Args:
            annotation_file: Path to the annotation JSON file

        Returns:
            List[Category]: List of all categories
        """
        data = self.load_coco_annotations(annotation_file)
        categories_data = data.get('categories', [])

        categories = []
        for cat_data in categories_data:
            category = Category(
                id=cat_data['id'],
                name=cat_data['name'],
                supercategory=cat_data['supercategory']
            )
            categories.append(category)

        logger.info(f"Loaded {len(categories)} categories")
        return categories

    def get_class_names(self, annotation_file: str) -> List[str]:
        """
        Get class names from COCO annotations.

        Args:
            annotation_file: Path to the annotation JSON file

        Returns:
            List[str]: List of class names
        """
        categories = self.get_categories(annotation_file)
        class_names = [cat.name for cat in categories]
        logger.info(f"Class names: {class_names}")
        return class_names

    def list_available_datasets(self) -> Dict[str, str]:
        """
        List all available datasets and their descriptions.

        Returns:
            Dict[str, str]: Dictionary of dataset names and descriptions
        """
        return {
            name: config["description"]
            for name, config in self.dataset_configs.items()
        }

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Optional[Dict]: Dataset configuration if found, None otherwise
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None

        return self.dataset_configs[dataset_name].copy()


def main() -> None:
    """Example usage of DatasetManager."""
    manager = DatasetManager()

    print("Available datasets:")
    for name, description in manager.list_available_datasets().items():
        print(f"  - {name}: {description}")

    # Example: Download COCO 2017 validation annotations
    dataset_path = manager.download_dataset("coco_2017_val")
    if dataset_path:
        print(f"Dataset downloaded to: {dataset_path}")

        # Load and display categories
        categories = manager.get_categories("annotations/instances_val2017.json")
        print(f"\nFirst 10 categories:")
        for cat in categories[:10]:
            print(f"  {cat.id}: {cat.name} ({cat.supercategory})")

        # Get class names
        class_names = manager.get_class_names("annotations/instances_val2017.json")
        print(f"\nClass names: {class_names[:10]}...")
    else:
        print("Failed to download dataset")


if __name__ == "__main__":
    main()