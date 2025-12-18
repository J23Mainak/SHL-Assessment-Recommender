import pandas as pd
import os

INPUT_XLSX = "Gen_AI Dataset.xlsx"
OUTPUT_DIR = "data"

def convert_dataset():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_XLSX):
        print(f"-> Error: {INPUT_XLSX} not found")
        return
    
    print(f"Loading {INPUT_XLSX}...")
    xls = pd.ExcelFile(INPUT_XLSX)
    
    print(f"Found sheets: {xls.sheet_names}")
    
    train_sheet = None
    test_sheet = None
    
    for sheet in xls.sheet_names:
        sheet_lower = sheet.lower()
        if 'train' in sheet_lower:
            train_sheet = sheet
        elif 'test' in sheet_lower:
            test_sheet = sheet
    
    # Fallback to first two sheets
    if not train_sheet and len(xls.sheet_names) > 0:
        train_sheet = xls.sheet_names[0]
    if not test_sheet and len(xls.sheet_names) > 1:
        test_sheet = xls.sheet_names[1]
    
    # Load and save train set
    if train_sheet:
        print(f"\nProcessing train set from sheet: {train_sheet}")
        train_df = xls.parse(train_sheet)
        
        # Normalize column names
        train_df.columns = [c.strip() for c in train_df.columns]
        
        # Save
        train_path = os.path.join(OUTPUT_DIR, "train_set.csv")
        train_df.to_csv(train_path, index=False)
        print(f"-> Saved train set: {train_path}")
        print(f"   Rows: {len(train_df)}")
        print(f"   Columns: {', '.join(train_df.columns.tolist())}")
        
        # Samples
        if 'Query' in train_df.columns:
            unique_queries = train_df['Query'].nunique()
            print(f"   Unique queries: {unique_queries}")
    
    # Load and save test set
    if test_sheet:
        print(f"\nProcessing test set from sheet: {test_sheet}")
        test_df = xls.parse(test_sheet)
        
        # Normalize column names
        test_df.columns = [c.strip() for c in test_df.columns]
        
        # Save
        test_path = os.path.join(OUTPUT_DIR, "test_set.csv")
        test_df.to_csv(test_path, index=False)
        print(f"-> Saved test set: {test_path}")
        print(f"   Rows: {len(test_df)}")
        print(f"   Columns: {', '.join(test_df.columns.tolist())}")
        
        if 'Query' in test_df.columns:
            unique_queries = test_df['Query'].nunique()
            print(f"   Unique queries: {unique_queries}")
    
    print("\nDataset conversion complete!")

if __name__ == "__main__":
    convert_dataset()