"""
Main orchestrator for the Road Hazard Detection System.
Coordinates data acquisition, feature extraction, model training, and evaluation.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import HAZARD_CLASSES, DATA_DIR, MODELS_DIR
from src.data_acquisition import DataSynchronizer
from src.feature_extraction import DataFrameFeatureExtractor, VisualFeatureExtractor
from src.model_trainer import RoadHazardClassifier, ModelComparator
from src.evaluator import EvaluationReporter, ThresholdAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_dataset(n_samples: int = 200) -> pd.DataFrame:
    """
    Create synthetic dataset for testing.
    In production, this would fetch real data from TxDOT and NOAA APIs.
    """
    logger.info(f"Creating synthetic dataset with {n_samples} samples...")

    # Synthetic data
    data = {
        'timestamp': pd.date_range('2026-01-01', periods=n_samples, freq='5min'),
        'location': np.random.choice(['austin_north', 'austin_south', 'san_marcos'], n_samples),
        'camera_id': [f'cam_{i}' for i in range(n_samples)],
        'image_url': [f'img_{i}.jpg' for i in range(n_samples)],
        # Visual features (simulated)
        'visual_feature_0': np.random.rand(n_samples),
        'visual_feature_1': np.random.rand(n_samples),
        'visual_feature_2': np.random.rand(n_samples),
        'visual_feature_3': np.random.rand(n_samples),
        'visual_feature_4': np.random.rand(n_samples),
        # Weather features
        'dewpoint': np.random.uniform(0, 30, n_samples),
        'relativeHumidity': np.random.uniform(20, 100, n_samples),
        'barometricPressure': np.random.uniform(1000, 1040, n_samples),
        'temperature': np.random.uniform(-5, 40, n_samples),
        'windSpeed': np.random.uniform(0, 50, n_samples),
        'visibility': np.random.uniform(0.1, 20, n_samples),
    }

    df = pd.DataFrame(data)

    # Create labels based on conditions
    labels = []
    for idx, row in df.iterrows():
        if row['windSpeed'] > 40:
            label = 4  # storm_risk
        elif row['temperature'] < 0 and row['relativeHumidity'] > 80:
            label = 3  # icy
        elif row['relativeHumidity'] > 95 and row['visibility'] < 2:
            label = 4  # storm_risk
        elif row['relativeHumidity'] > 70:
            label = 1  # wet
        elif row['temperature'] < -5:
            label = 2  # snowy
        else:
            label = 0  # safe

        labels.append(label)

    df['hazard_class'] = labels
    df['hazard_label'] = df['hazard_class'].map(HAZARD_CLASSES)

    logger.info(f"Synthetic dataset created: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Class distribution:\n{df['hazard_label'].value_counts()}")

    return df


def run_pipeline():
    """Execute the complete road hazard detection pipeline."""
    logger.info("=" * 80)
    logger.info("ROAD HAZARD DETECTION SYSTEM - PIPELINE EXECUTION")
    logger.info("=" * 80)

    # Step 1: Data Acquisition
    logger.info("\n[1/5] DATA ACQUISITION & SYNCHRONIZATION")
    logger.info("-" * 80)

    try:
        # In production, use real data from DataSynchronizer
        # synchronizer = DataSynchronizer()
        # df = synchronizer.fetch_all_locations()
        # synchronizer.save_synchronized_data(df)

        # For now, use synthetic data for demonstration
        df = create_synthetic_dataset(n_samples=300)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(os.path.join(DATA_DIR, 'synchronized_data.csv'), index=False)
        logger.info(f"✓ Data acquisition complete: {df.shape[0]} samples")

    except Exception as e:
        logger.error(f"✗ Data acquisition failed: {e}")
        return

    # Step 2: Feature Engineering
    logger.info("\n[2/5] FEATURE ENGINEERING")
    logger.info("-" * 80)

    try:
        # In production, extract visual features from images
        # extractor = DataFrameFeatureExtractor()
        # df = extractor.extract_visual_features_for_dataframe(df, 'image_url')

        # For now, features are already in synthetic data
        logger.info("✓ Feature extraction complete")

        # Separate visual and weather features
        visual_features = [col for col in df.columns if col.startswith('visual_feature_')]
        weather_features = [
            'dewpoint', 'relativeHumidity', 'barometricPressure',
            'temperature', 'windSpeed', 'visibility'
        ]

        logger.info(f"  Visual features: {len(visual_features)}")
        logger.info(f"  Weather features: {len(weather_features)}")

    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        return

    # Step 3: Model Training & Comparison
    logger.info("\n[3/5] MODEL TRAINING & COMPARISON")
    logger.info("-" * 80)

    try:
        # Prepare data
        X_visual = df[visual_features].values
        X_weather = df[weather_features].values
        X_fused = np.concatenate([X_visual, X_weather], axis=1)
        y = df['hazard_class'].values

        all_features = visual_features + weather_features

        # Compare models
        comparator = ModelComparator()
        comparison_results = comparator.compare_models(
            X_fused, X_visual, y,
            feature_names_fused=all_features,
            feature_names_camera=visual_features
        )

        logger.info("✓ Model training and comparison complete")
        comparator.print_comparison_report()

    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        return

    # Step 4: Feature Importance & Threshold Analysis
    logger.info("\n[4/5] FEATURE IMPORTANCE & THRESHOLD ANALYSIS")
    logger.info("-" * 80)

    try:
        # Get feature importance
        importance_df = comparator.fused_model.get_feature_importance(top_n=20)

        # Get probabilities for threshold analysis
        X_fused_scaled = comparator.fused_model.scaler.transform(X_fused)
        y_pred_proba = comparator.fused_model.model.predict_proba(X_fused_scaled)

        # Compute thresholds
        analyzer = ThresholdAnalyzer()
        thresholds = analyzer.compute_confidence_thresholds(y, y_pred_proba, target_precision=0.95)

        logger.info("✓ Feature importance and threshold analysis complete")
        logger.info(f"  Computed {len(thresholds)} class-specific thresholds")

    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}")
        return

    # Step 5: Evaluation & Reporting
    logger.info("\n[5/5] EVALUATION & REPORTING")
    logger.info("-" * 80)

    try:
        # Make predictions
        y_fused_pred = comparator.fused_model.predict(X_fused)
        y_camera_pred = comparator.camera_only_model.predict(X_visual)

        # Generate reports
        reporter = EvaluationReporter()
        reporter.generate_classification_report(y, y_fused_pred, "fused_model")
        reporter.generate_classification_report(y, y_camera_pred, "camera_only")

        # Plot visualizations
        reporter.plot_confusion_matrix(y, y_fused_pred, "fused_model")
        reporter.plot_confusion_matrix(y, y_camera_pred, "camera_only")
        reporter.plot_feature_importance(importance_df, "fused_model")
        reporter.plot_model_comparison(comparison_results)

        # Save models
        os.makedirs(MODELS_DIR, exist_ok=True)
        comparator.fused_model.save_model(os.path.join(MODELS_DIR, 'fused_model.pkl'))
        comparator.camera_only_model.save_model(os.path.join(MODELS_DIR, 'camera_only_model.pkl'))

        # Generate summary report
        reporter.generate_summary_report(
            comparison_results['fused_model'],
            comparison_results['camera_only_model'],
            comparison_results['improvement'],
            importance_df
        )

        logger.info("✓ Evaluation and reporting complete")

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        return

    # Success summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nGenerated outputs:")
    logger.info(f"  Data: {DATA_DIR}/synchronized_data.csv")
    logger.info(f"  Models: {MODELS_DIR}/")
    logger.info(f"  Reports: results/")
    logger.info("\nNext steps for git workflow:")
    logger.info("  git add .")
    logger.info("  git commit -m 'feat: initial data acquisition and model training'")
    logger.info("  git push")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    run_pipeline()
