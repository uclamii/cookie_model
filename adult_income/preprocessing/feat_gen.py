################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import pandas as pd
import numpy as np
import typer

from adult_income.functions import mlflow_dumpArtifact, mlflow_loadArtifact

from adult_income.constants import (
    var_index,
    exp_artifact_name,
    preproc_run_name,
    target_outcome,
)

################################################################################
################ Model Preprocessing and Feature Engineering ###################
################################################################################

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer()

################################################################################
################ Step 3: Define Main Function ##################################
################################################################################


@app.command()
def main(
    input_data_file: str = "./data/processed/df_sans_zero_missing.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Processes the input data file and generates feature space X and target variable y.

    Args:
        input_data_file (str): Path to the input parquet file.
    """

    ############################################################################
    ################ Step 4: Load Input Data ###################################
    ############################################################################

    # Read the input data file
    df = pd.read_parquet(input_data_file)

    try:
        df.set_index(var_index)
    except:
        print("Index already set or 'var_index' doesn't exist in dataframe")
    print("-" * 80)
    print(f"# of DataFrame Columns: {df.shape[1]}")

    ############################################################################
    ################ Step 5: Training Stage ####################################
    ############################################################################

    if stage == "training":

        ############### Store Final List of Features for Production ##

        # Step 1: df already loaded from .parquet
        # Example:
        # df = pd.read_parquet("path_to_file.parquet")

        # Step 2: Separate features (X) and target (y)
        X = df.drop(columns=["income"]).copy()
        y = df[["income"]].copy()  # keep as DataFrame for now

        # Step 4: Log first five rows of features and targets
        print(f"\n{'=' * 80}\nX\n{'=' * 80}\n{X.head()}")
        print(f"\n{'=' * 80}\ny\n{'=' * 80}\n{y.head()}")

        # Step 5: Retain numeric columns only
        X = X.select_dtypes(include=np.number)

        # Clean target column by removing trailing period
        y.loc[:, "income"] = y["income"].str.rstrip(".")

        # Step 9: Encode target to binary
        y = y["income"].map({"<=50K": 0, ">50K": 1})

        # Step 8: Display class balance
        print(f"\nBreakdown of y:\n{y.value_counts()}\n")
        print(y)

        X_columns_list = X.columns.to_list()

    ############################################################################
    ################ Step 6: Inference Stage Load X_columns list ###############
    ############################################################################

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Features List From `feat_gen.py`
        ########################################################################
        # During training, we identified and stored `X_columns_list`.
        # Now, we reload this to ensure that inference follows the same
        # preprocessing pipeline as training, maintaining consistency.
        ########################################################################

        # Load feature column names from Mlflow
        X_columns_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Use the same run_name as training
            obj_name="X_columns_list",
        )

        X = df[X_columns_list].copy()

        # dropping "_missing" cols strings will be created next
        X_columns_list = [col for col in X_columns_list if "_missing" not in col]

    ############################################################################
    ################ Step 7: Create and Append Missingness Indicators ##########
    ############################################################################

    # Create missing indicators for each feature
    X_missing = X.isna().astype(int)
    X_missing.columns = [f"{col}_missing" for col in X.columns]

    ############################################################################
    ################ Step 8: Store Final List of Features for Production #######
    ############################################################################
    if stage == "training":
        # Save feature column names to a pickle file
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,  # Consistent run_name for all artifacts
            obj_name="X_columns_list",
            obj=X_columns_list,
        )
        print(f"\nShape of X: {X.shape} \n")

    # Append missing indicators to feature space
    X = pd.concat([X, X_missing], axis=1)

    print(f"New X Shape (after adding missingness indicators): {X.shape}")
    if stage == "inference":
        print(
            "\033[33mNumber of rows may vary due to Step 17 of `preprocessing.py`\033[0m"
        )
    print("-" * 80)
    print(f"\nFeature Space\n{X.head()}\n")

    ############################################################################
    ################ Step 10: Generate Target Variable for Training #############
    ############################################################################

    if stage == "training":
        # Target variables from constants.py
        y = pd.DataFrame(y)
        y.to_parquet(
            os.path.join(data_path, f"y_{target_outcome}.parquet"),
        )

    ############################################################################
    ################ Step 11: Save Processed Feature Space #####################
    ############################################################################

    # Save the feature space (X) and target variables (y) to parquet files
    X.to_parquet(os.path.join(data_path, "X.parquet"))


################################################################################

if __name__ == "__main__":
    app()
