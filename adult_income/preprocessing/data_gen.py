#!/usr/bin/env python3
"""
data_gen.py
-----------
Fetches the UCI Adult Income dataset using ucimlrepo, saves both CSV and
Parquet files under data/raw for reproducibility and downstream processing.

Example usage:
    python data_gen.py --output-data-file ./data/raw/df.parquet
"""

import os
import typer
import pandas as pd
from ucimlrepo import fetch_ucirepo

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_data_file: str = "./data/raw/df.parquet",
    csv_backup: bool = True,
):
    """
    Fetch the UCI Adult Income dataset and save to Parquet and optional CSV.

    Parameters
    ----------
    output_data_file : str
        Path where the Parquet file will be saved.
    csv_backup : bool, optional
        Whether to also save a CSV backup in the same directory.
    """

    typer.echo("Fetching Adult Income dataset from UCI ML Repository...")

    dataset = fetch_ucirepo(id=2)  ## Fetch dataset
    df = dataset.data.features.join(dataset.data.targets, how="inner")

    ## Ensure the output directory exists
    os.makedirs(os.path.dirname(output_data_file), exist_ok=True)

    ## Save Parquet file
    df.to_parquet(output_data_file, index=False)
    typer.echo(f"Saved dataset to {output_data_file}")

    ## Optionally save CSV backup
    if csv_backup:
        csv_path = os.path.splitext(output_data_file)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        typer.echo(f"Saved CSV backup to {csv_path}")

    typer.echo("Data generation complete.")


if __name__ == "__main__":
    app()
