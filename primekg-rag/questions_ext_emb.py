import pandas as pd
import sys

# --- Configuration ---
EXCEL_FILE_PATH = "C://Users//aemekkawi//Documents//GitHub//primekg-rag//primekg-rag//Codebook opzet.xlsx"  # <-- IMPORTANT: Change this to your actual Excel file name
SHEET_NAME = "MINI"
QUESTION_COLUMN_NAME = "Item + question"   
OUTPUT_CSV_FILE = "questions_for_mapping.csv"

def extract_questions_from_excel():
    """
    Reads an Excel sheet, extracts a specific column, and saves it to a CSV file.
    """
    print(f"--- Starting Question Extraction ---")
    
    try:
        # Read the specific sheet from the Excel file
        print(f"Reading sheet '{SHEET_NAME}' from '{EXCEL_FILE_PATH}'...")
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME)
        
    except FileNotFoundError:
        print(f"ERROR: The file '{EXCEL_FILE_PATH}' was not found. Please make sure it's in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not read the Excel file. Reason: {e}")
        sys.exit(1)

    # Check if the question column exists
    if QUESTION_COLUMN_NAME not in df.columns:
        print(f"ERROR: The column '{QUESTION_COLUMN_NAME}' was not found in the '{SHEET_NAME}' sheet.")
        print(f"Available columns are: {list(df.columns)}")
        sys.exit(1)

    # Extract the questions and remove any empty rows
    questions = df[[QUESTION_COLUMN_NAME]].dropna()
    questions.rename(columns={QUESTION_COLUMN_NAME: 'question_text'}, inplace=True)

    # Save the extracted questions to a new CSV file
    try:
        questions.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Successfully extracted {len(questions)} questions.")
        print(f"Data saved to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"ERROR: Could not save the CSV file. Reason: {e}")

if __name__ == "__main__":
    extract_questions_from_excel()