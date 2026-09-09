import json
import os

files = [
    r"d:\NTI\Project\customer-churn-prediction\Telco_Customer_Churn_Prediction.ipynb",
    r"d:\NTI\Project\customer-churn-prediction\Project.ipynb"
]

target_str = "X = df.drop(columns=['Churn', 'gender', 'PhoneService'])\n"
replacement_str = "X = df.drop(columns=['Churn', 'gender', 'PhoneService', 'TotalCharges'])\n"

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    updated = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if target_str in line:
                    source[i] = line.replace(target_str, replacement_str)
                    updated = True
                    
    if updated:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        print(f"Updated {file}")
    else:
        print(f"Target string not found in {file}")
