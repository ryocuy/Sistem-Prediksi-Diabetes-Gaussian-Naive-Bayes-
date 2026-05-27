import openpyxl
import csv

wb = openpyxl.load_workbook('/Users/riowoy/Desktop/tugas smester4/kecerdasan buatan /diabetes/data_ujian_akhir.xlsx', data_only=True)
ws = wb['data_4']

# Header: Age, BMI, Glucose, Insulin, Classification
rows = []
for i, r in enumerate(ws.iter_rows(min_row=2, values_only=True)):
    if r[0] is None: continue
    age, bmi, glucose, insulin, cls = r
    
    # Convert classification: '1' -> 0 (Sehat), '2' -> 1 (Sakit)
    outcome = 0 if str(cls) == '1' else 1
    rows.append({
        'Glucose': glucose,
        'BMI': bmi,
        'Age': age,
        'Outcome': outcome
    })

# Sort rows: Outcome 1 first, then Outcome 0
rows.sort(key=lambda x: x['Outcome'], reverse=True)

with open('/Users/riowoy/Desktop/tugas smester4/kecerdasan buatan /diabetes/diabetes.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Glucose', 'BMI', 'Age', 'Outcome'])
    writer.writeheader()
    writer.writerows(rows)

print("Dataset prepared and saved to diabetes.csv")
