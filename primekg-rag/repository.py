import pandas as pd
import os
from tqdm import tqdm  # Assuming tqdm is used for progress bars

# --- Configuration (Adjust paths if necessary) ---
INPUT_KG_CSV_PATH = "primekg-rag/kg.csv"  # Make sure this path is correct
OUTPUT_VERBALIZED_PARQUET_PATH = "primekg-rag/kg_verbalized_prepared.parquet"
# --- End Configuration ---


def verbalize_kg_row(row):
    # This function should create a natural language sentence from a KG triple
    # Ensure this logic correctly uses the column names from your kg.csv
    # Example (adjust according to your kg.csv columns):
    subject = row["x_name"] if "x_name" in row else row["x_id"]
    relation = row["display_relation"] if "display_relation" in row else row["relation"]
    obj = row["y_name"] if "y_name" in row else row["y_id"]
    return f"{subject} {relation} {obj}."


def create_metadata(row):
    # This function should create a metadata dictionary for each row
    # Example:
    metadata = {
        "x_id": row.get("x_id"),
        "x_name": row.get("x_name"),
        "x_type": row.get("x_type"),
        "relation": row.get("relation"),
        "display_relation": row.get("display_relation"),
        "y_id": row.get("y_id"),
        "y_name": row.get("y_name"),
        "y_type": row.get("y_type"),
        "x_source": row.get("x_source"),
        "y_source": row.get("y_source"),
    }
    # Filter out None values to keep metadata clean
    return {k: v for k, v in metadata.items() if v is not None}


def prepare_verbalized_data(input_csv_path, output_parquet_path):
    print(f"--- Starting data preparation from {input_csv_path} ---")

    try:
        print(f"Loading data from {input_csv_path}...")
        df_kg = pd.read_csv(input_csv_path)
        print(f"Loaded {len(df_kg)} rows from CSV.")
        print(f"Initial columns in DataFrame: {df_kg.columns.tolist()}")

        if df_kg.empty:
            print("WARNING: The loaded CSV is empty. No verbalized data will be generated.")
            # Create an empty DataFrame with expected columns to avoid errors
            # later
            df_verbalized = pd.DataFrame(columns=["verbalized_text", "metadata"])
        else:
            print("Generating verbalized text...")
            # Use .progress_apply if tqdm is installed and configured for
            # pandas
            df_kg["verbalized_text"] = df_kg.progress_apply(verbalize_kg_row, axis=1)
            print(f"Verbalized text generated. Current columns: {df_kg.columns.tolist()}")

            print("Generating metadata...")
            df_kg["metadata"] = df_kg.progress_apply(create_metadata, axis=1)
            print(f"Metadata generated. Current columns: {df_kg.columns.tolist()}")

            # Select only the columns needed for the output Parquet file
            df_verbalized = df_kg[["verbalized_text", "metadata"]]
            print(f"Selected final columns for Parquet. Final DataFrame columns: {df_verbalized.columns.tolist()}")

        print(f"Saving prepared data to {output_parquet_path}...")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        df_verbalized.to_parquet(output_parquet_path, index=False)
        print("Data preparation complete. Verbalized data saved to Parquet.")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Ensure tqdm is hooked into pandas if you want progress bars
    try:
        from tqdm.auto import tqdm as tqdm_auto

        tqdm_auto.pandas()
    except ImportError:
        print("tqdm.auto not found, progress bars will not be shown.")

    prepare_verbalized_data(INPUT_KG_CSV_PATH, OUTPUT_VERBALIZED_PARQUET_PATH)
