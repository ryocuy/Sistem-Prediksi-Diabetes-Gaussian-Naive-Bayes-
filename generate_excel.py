import csv
import openpyxl

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Hitungan"

# 1. Tulis Dataset ke Kolom A-I
with open("diabetes.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row_idx, row in enumerate(reader, 1):
        for col_idx, val in enumerate(row, 1):
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass
            ws.cell(row=row_idx, column=col_idx, value=val)

# 2. Setup Layout Hitungan di sebelah kanan
ws['J1'] = "Naive Bayes Gaussian - Dataset Diabetes"

ws['J3'] = "Kelas"
ws['K3'] = "Jumlah"
ws['L3'] = "P(H)"
ws['J4'] = 1; ws['K4'] = 50; ws['L4'] = "=K4/100"
ws['J5'] = 0; ws['K5'] = 50; ws['L5'] = "=K5/100"

ws['J7'] = "Kelas 1"
ws['J8'] = "Rata-rata"; ws['K8'] = "Glucose"; ws['L8'] = "BMI"; ws['M8'] = "Age"
ws['K9'] = "=AVERAGE(A2:A51)"; ws['L9'] = "=AVERAGE(B2:B51)"; ws['M9'] = "=AVERAGE(C2:C51)"
ws['J10'] = "Varian"; ws['K10'] = "=VAR.S(A2:A51)"; ws['L10'] = "=VAR.S(B2:B51)"; ws['M10'] = "=VAR.S(C2:C51)"

ws['J12'] = "Kelas 0"
ws['J13'] = "Rata-rata"; ws['K13'] = "=AVERAGE(A52:A101)"; ws['L13'] = "=AVERAGE(B52:B101)"; ws['M13'] = "=AVERAGE(C52:C101)"
ws['J14'] = "Varian"; ws['K14'] = "=VAR.S(A52:A101)"; ws['L14'] = "=VAR.S(B52:B101)"; ws['M14'] = "=VAR.S(C52:C101)"

ws['K16'] = "Nilai"; ws['L16'] = "Kelas 1"; ws['M16'] = "Kelas 0"

# FIXED: Use -1 * (x - mean)^2 to avoid Excel unary minus bug
ws['J17'] = "Glucose"; ws['K17'] = 120
ws['L17'] = "=1/SQRT(2*3.14*K10)*(2.72^(-1*((K17-K9)^2)/(2*K10)))"
ws['M17'] = "=1/SQRT(2*3.14*K14)*(2.72^(-1*((K17-K13)^2)/(2*K14)))"

ws['J18'] = "BMI"; ws['K18'] = 25
ws['L18'] = "=1/SQRT(2*3.14*L10)*(2.72^(-1*((K18-L9)^2)/(2*L10)))"
ws['M18'] = "=1/SQRT(2*3.14*L14)*(2.72^(-1*((K18-L13)^2)/(2*L14)))"

ws['J19'] = "Age"; ws['K19'] = 30
ws['L19'] = "=1/SQRT(2*3.14*M10)*(2.72^(-1*((K19-M9)^2)/(2*M10)))"
ws['M19'] = "=1/SQRT(2*3.14*M14)*(2.72^(-1*((K19-M13)^2)/(2*M14)))"

ws['J20'] = "output?"; ws['K20'] = 1
ws['J21'] = "Akhir Kali"; ws['K21'] = "Kelas 1"; ws['L21'] = "Kelas 0"
ws['K22'] = "=L4*L17*L18*L19"
ws['L22'] = "=L5*M17*M18*M19"

ws['J23'] = "Hasil Prediksi"
ws['K23'] = '=IF(K22>L22, "Kelas 1", "Kelas 0")'
ws['J24'] = "cari nilai probabilitas tertinggi"
ws['K24'] = '=IF(K22>L22, "Kelas 1 (Sakit)", "Kelas 0 (Negatif/Sehat)")'

wb.save("Hitungan_UAS_Naive_Bayes.xlsx")
print("File Hitungan_UAS_Naive_Bayes.xlsx diperbaiki!")
