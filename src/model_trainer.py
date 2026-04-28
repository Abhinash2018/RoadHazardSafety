"""
Model training module for Random Forest classifier.
Trains on fused visual + atmospheric data and compares against camera-only baseline.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import pickle
import logging
import os
from typing import Dict, Tuple, Optional

from config import RANDOM_FOREST_PARAMS, CV_FOLDS, TEST_SPLIT, MODELS_DIR, HAZARD_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoadHazardClassifier:
    """Random Forest classifier for road hazard detection."""

    def __init__(self, params: Dict = None):
        """Initialize classifier with parameters."""
        if params is None:
            params = RANDOM_FOREST_PARAMS

        self.params = params
        self.model = RandomForestClassifier(**params)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def train(
        self, X: np.ndarray, y: np.ndarray, feature_names: list = None
    ) -> Dict:
        """
        Train the Random Forest classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            feature_names: Optional list of feature names

        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info(f"Training classifier on {X.shape[0]} samples with {X.shape[1]} features")

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self.feature_names = feature_names

            # Training metrics
            train_pred = self.model.predict(X_scaled)
            metrics = {
                "accuracy": accuracy_score(y, train_pred),
                "f1": f1_score(y, train_pred, average="weighted"),
                "precision": precision_score(y, train_pred, average="weighted"),
                "recall": recall_score(y, train_pred, average="weighted"),
            }

            logger.info(f"Training metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise

    def evaluate_with_cv(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate using cross-validation.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with CV metrics
        """
        try:
            X_scaled = self.scaler.fit_transform(X)

            # Perform cross-validation
            cv_scores_f1 = cross_val_score(
                self.model, X_scaled, y, cv=CV_FOLDS, scoring="f1_weighted"
            )
            cv_scores_acc = cross_val_score(
                self.model, X_scaled, y, cv=CV_FOLDS, scoring="accuracy"
            )

            metrics = {
                "cv_f1_mean": np.mean(cv_scores_f1),
                "cv_f1_std": np.std(cv_scores_f1),
                "cv_accuracy_mean": np.mean(cv_scores_acc),
                "cv_accuracy_std": np.std(cv_scores_acc),
            }

            logger.info(f"Cross-validation metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance ranking."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importances = self.model.feature_importances_
        
        # Ensure we don't try to get more features than available
        top_n = min(top_n, len(importances))
        
        indices = np.argsort(importances)[::-1][:top_n]

        # Create feature names list safely
        if self.feature_names is None or len(self.feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        feature_importance_df = pd.DataFrame(
            {
                "rank": range(1, top_n + 1),
                "feature": [feature_names[i] for i in indices],
                "importance": importances[indices],
            }
        )

        logger.info(f"\nTop {top_n} Feature Importances:")
        logger.info(feature_importance_df.to_string(index=False))
        return feature_importance_df

    def save_model(self, filepath: str) -> bool:
        """Save trained model to disk."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(
                    {"model": self.model, "scaler": self.scaler, "feature_names": self.feature_names},
                    f,
                )
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load trained model from disk."""
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.feature_names = data["feature_names"]
                self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class ModelComparator:
    """Compares fused model vs camera-only baseline."""

    def __init__(self):
        self.fused_model = RoadHazardClassifier()
        self.camera_only_model = RoadHazardClassifier()
        self.results = {}

    def identify_visual_features(self, all_features: list) -> list:
        """Identify which features are visual (camera-based)."""
        # Assuming features starting with 'visual_' are camera features
        return [f for f in all_features if f.startswith("visual_feature_")]

    def identify_weather_features(self, all_features: list) -> list:
        """Identify which features are weather-based (atmospheric)."""
        weather_keywords = [
            "dewpoint",
            "relativeHumidity",
            "barometricPressure",
            "temperature",
            "windSpeed",
            "visibility",
        ]
        return [
            f for f in all_features if any(keyword in f for keyword in weather_keywords)
        ]

    def compare_models(
        self, X_fused: np.ndarray, X_camera_only: np.ndarray, y: np.ndarray,
        feature_names_fused: list = None, feature_names_camera: list = None
    ) -> Dict:
        """
        Compare fused model against camera-only baseline.

        Args:
            X_fused: Fused feature matrix
            X_camera_only: Camera-only feature matrix
            y: Target labels
            feature_names_fused: Feature names for fused model
            feature_names_camera: Feature names for camera model

        Returns:
            Comparison results dictionary
        """
        logger.info("Comparing Fused Model vs Camera-Only Baseline...")

        # Split data
        X_fused_train, X_fused_test, y_train, y_test = train_test_split(
            X_fused, y, test_size=TEST_SPLIT, random_state=42
        )

        X_camera_train, X_camera_test, _, _ = train_test_split(
            X_camera_only, y, test_size=TEST_SPLIT, random_state=42
        )

        # Train both models
        logger.info("Training fused model...")
        self.fused_model.train(X_fused_train, y_train, feature_names_fused)

        logger.info("Training camera-only baseline...")
        self.camera_only_model.train(X_camera_train, y_train, feature_names_camera)

        # Evaluate on test set
        fused_pred = self.fused_model.predict(X_fused_test)
        camera_pred = self.camera_only_model.predict(X_camera_test)

        fused_metrics = {
            "accuracy": accuracy_score(y_test, fused_pred),
            "f1": f1_score(y_test, fused_pred, average="weighted"),
            "precision": precision_score(y_test, fused_pred, average="weighted"),
            "recall": recall_score(y_test, fused_pred, average="weighted"),
        }

        camera_metrics = {
            "accuracy": accuracy_score(y_test, camera_pred),
            "f1": f1_score(y_test, camera_pred, average="weighted"),
            "precision": precision_score(y_test, camera_pred, average="weighted"),
            "recall": recall_score(y_test, camera_pred, average="weighted"),
        }

        # Calculate improvement
        improvement = {
            "accuracy_improvement": (
                (fused_metrics["accuracy"] - camera_metrics["accuracy"])
                / camera_metrics["accuracy"]
                * 100
            ),
            "f1_improvement": (
                (fused_metrics["f1"] - camera_metrics["f1"]) / camera_metrics["f1"] * 100
            ),
        }

        self.results = {
            "fused_model": fused_metrics,
            "camera_only_model": camera_metrics,
            "improvement": improvement,
            "test_size": len(y_test),
        }

        logger.info(f"\nFused Model Metrics: {fused_metrics}")
        logger.info(f"Camera-Only Model Metrics: {camera_metrics}")
        logger.info(f"Improvement: {improvement}")

        return self.results

    def print_comparison_report(self):
        """Print detailed comparison report."""
        if not self.results:
            logger.warning("No comparison results available")
            return

        print("\n" + "=" * 80)
        print("MODEL COMPARISON REPORT: Fused vs Camera-Only")
        print("=" * 80)

        print("\nFUSED MODEL (Visual + Atmospheric):")
        for metric, value in self.results["fused_model"].items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nCAMERA-ONLY BASELINE (Visual only):")
        for metric, value in self.results["camera_only_model"].items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nFUSION ADVANTAGE:")
        for metric, value in self.results["improvement"].items():
            sign = "+" if value > 0 else ""
            print(f"  {metric.upper()}: {sign}{value:.2f}%")

        print(f"\nTest set size: {self.results['test_size']} samples")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    logger.info("Model trainer module loaded")
