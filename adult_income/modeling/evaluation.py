from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import numpy as np

# Import necessary modules from custom libraries
from model_tuner import find_optimal_threshold_beta

# Import supportive care functions and constants
from adult_income.functions import (
    mlflow_load_model,
    return_model_metrics,
    return_model_plots,
    log_mlflow_metrics,
    mlflow_log_parameters_model,
)

from adult_income.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
    target_precision,
    threshold_target_metric,
)

app = typer.Typer()

################################################################################
# ---- STEP 1: Define command-line arguments with default values ----
################################################################################


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_type: str = "lr",
    pipeline_type: str = "orig",
    outcome: str = "ISDEATHDATElead6mo",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_ISDEATHDATElead1yr.parquet",
    scoring: str = "average_precision",
    # -----------------------------------------
):

    ################################################################################
    # STEP 2: Load Model Configuration & Pipeline Settings
    ################################################################################

    estimator_name = model_definitions[model_type]["estimator_name"]

    print(f"{estimator_name}_{pipeline_type}_training")
    print(f"{estimator_name}_{outcome}")

    ################################################################################
    # STEP 3: Load Pre-Trained Model from MLflow
    ################################################################################

    model = mlflow_load_model(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}",
    )

    # Print model threshold before optimization
    print(f"Model Threshold Before Threshold Optimization: {model.threshold}")

    ################################################################################
    # STEP 4: Load Processed Data (Features & Labels)
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)
    y = y.squeeze()  # coerce into a series
    ################################################################################
    # STEP 5: Split Data into Train, Validation, and Test Sets
    ################################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)

    ################################################################################
    # STEP 6: Find the Optimal Threshold Based on Target Precision
    # (defined in config.py under `target_precision`)
    ################################################################################

    try:
        print(f"\n Best Threshold and Beta For {outcome}: \n")
        threshold, beta = find_optimal_threshold_beta(
            y_valid,
            model.predict_proba(X_valid)[:, 1],
            target_metric=threshold_target_metric,
            target_score=target_precision,
            beta_value_range=np.linspace(0.01, 4, 40),
            delta=0.01,
        )

        # Store optimized threshold under the appropriate scoring metric
        model.threshold[scoring] = threshold
        model.beta = beta

    except Exception as e:
        print(
            f"Could not find optimal threshold for the {threshold_target_metric} of {target_precision}"
        )
        print(e)

    ################################################################################
    # STEP 7: Log Updated Model with Optimized Threshold
    ################################################################################

    mlflow_log_parameters_model(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}",
        model=model,
    )

    # Print model threshold after optimization
    print(f"Model Threshold After Threshold Optimization: {model.threshold}")

    ################################################################################
    # STEP 8: Compute and Evaluate Model Performance Metrics
    ################################################################################

    all_inputs = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "valid": (X_valid, y_valid),
    }
    metrics = return_model_metrics(
        inputs=all_inputs,
        model=model,
        estimator_name=estimator_name,
    )

    print(metrics)

    ################################################################################
    # STEP 9: Generate and Save Model Evaluation Plots
    ################################################################################

    # Generate evaluation plots
    all_plots = return_model_plots(
        inputs=all_inputs,
        model=model,
        estimator_name=estimator_name,
        scoring=scoring,
    )

    ################################################################################
    # STEP 10: Log Experiment Details to MLflow
    ################################################################################

    log_mlflow_metrics(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        metrics=metrics[estimator_name],
        images=all_plots,
    )

    ################################################################################
    # STEP 11: Completion Message
    ################################################################################

    logger.success("Modeling evaluation complete.")
    # -----------------------------------------


if __name__ == "__main__":

    app()
