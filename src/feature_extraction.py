"""
Visual feature extraction module using OpenCV.
Extracts texture and appearance descriptors from traffic camera images.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class VisualFeatureExtractor:
    """Extracts visual features from road camera images."""

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize feature extractor.

        Args:
            target_size: Standard image size for feature extraction
        """
        self.target_size = target_size

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file."""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None

            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Resize to standard size
            image = cv2.resize(image, self.target_size)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def extract_color_histogram(
        self, image: np.ndarray, bins: int = 32
    ) -> np.ndarray:
        """
        Extract color histogram features from image.

        Args:
            image: Input image (BGR)
            bins: Number of histogram bins per channel

        Returns:
            Flattened histogram features (96 features for 3 channels)
        """
        hist_features = []

        # Extract histogram for each channel (B, G, R)
        for channel in range(3):
            hist = cv2.calcHist(
                [image], [channel], None, [bins], [0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)

        return np.array(hist_features)

    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract texture features using edge detection.

        Args:
            image: Input image

        Returns:
            Texture feature vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel edge detection (horizontal and vertical)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Compute edge magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)

        # Extract statistics as features
        texture_features = np.array(
            [
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude),
                np.mean(direction),
                np.std(direction),
            ]
        )

        return texture_features

    def extract_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Extract brightness and contrast features.

        Args:
            image: Input image

        Returns:
            Brightness and contrast features
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        contrast = np.std(gray)
        dynamic_range = np.max(gray) - np.min(gray)

        return np.array([brightness, contrast, dynamic_range])

    def extract_saturation_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color saturation features (useful for detecting wet/snowy roads).

        Args:
            image: Input image

        Returns:
            Saturation-based features
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract saturation channel
        saturation = hsv[:, :, 1]

        saturation_features = np.array(
            [
                np.mean(saturation),
                np.std(saturation),
                np.max(saturation),
                np.min(saturation),
            ]
        )

        return saturation_features

    def extract_all_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all visual features from an image.

        Args:
            image: Input image

        Returns:
            Dictionary of feature arrays
        """
        features = {
            "color_histogram": self.extract_color_histogram(image),
            "texture": self.extract_texture_features(image),
            "brightness_contrast": self.extract_brightness_contrast(image),
            "saturation": self.extract_saturation_features(image),
        }

        return features

    def flatten_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten feature dictionary into single vector."""
        return np.concatenate(list(features_dict.values()))

    def extract_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """Extract all features from image file."""
        image = self.load_image(image_path)
        if image is None:
            return None

        features = self.extract_all_features(image)
        return self.flatten_features(features)


class DataFrameFeatureExtractor:
    """Applies feature extraction to DataFrames with image paths."""

    def __init__(self):
        self.extractor = VisualFeatureExtractor()

    def extract_visual_features_for_dataframe(
        self, df: pd.DataFrame, image_column: str = "image_url"
    ) -> pd.DataFrame:
        """
        Extract visual features for all images in a DataFrame.

        Args:
            df: Input DataFrame with image column
            image_column: Name of column containing image paths/URLs

        Returns:
            DataFrame with added visual feature columns
        """
        # Create a copy to avoid modifying original
        result_df = df.copy()

        # Extract features for each image
        visual_features_list = []

        for idx, row in result_df.iterrows():
            image_path = row.get(image_column)

            if pd.isna(image_path) or image_path is None:
                logger.warning(f"Skipping row {idx}: No image path")
                visual_features_list.append(None)
                continue

            # Extract features
            features = self.extractor.extract_from_file(str(image_path))

            if features is not None:
                visual_features_list.append(features)
            else:
                logger.warning(f"Could not extract features for {image_path}")
                visual_features_list.append(None)

        # Create feature columns
        feature_names = [
            f"visual_feature_{i}" for i in range(len(visual_features_list[0]))
        ]

        # Only add rows where features were successfully extracted
        features_array = np.array(
            [f for f in visual_features_list if f is not None]
        )

        if len(features_array) > 0:
            for i, feature_name in enumerate(feature_names):
                result_df[feature_name] = np.nan
                # Fill in features where available
                valid_idx = 0
                for j, features in enumerate(visual_features_list):
                    if features is not None:
                        result_df.at[j, feature_name] = features[i]

            logger.info(f"Extracted visual features for {len(features_array)} images")

        return result_df


if __name__ == "__main__":
    # Example usage
    extractor = VisualFeatureExtractor()
    print(f"Feature extractor initialized with target size: {extractor.target_size}")
