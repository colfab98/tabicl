import pandas as pd

path = "/home/fcolanto/projects/pfn-augmentation/data/Datacortech_final_data.xlsx"
xls = pd.ExcelFile(path)

print("Sheets:", xls.sheet_names)
for sheet in xls.sheet_names:
    df = pd.read_excel(path, sheet_name=sheet)
    print(f"\n=== SHEET: {sheet} ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(3).to_string())
    print("\ndtypes:")
    print(df.dtypes.to_string())
