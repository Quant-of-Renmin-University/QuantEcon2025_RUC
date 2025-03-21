import os
from docxtpl import DocxTemplate
import pandas as pd
from docx2pdf import convert

# Define paths
TEMPLATE_PATH = r"C:\桌面文件下载在这里\template.docx"
EXCEL_PATH = r"C:\桌面文件下载在这里\programs.xlsx"
OUTPUT_DIR = r"C:\桌面文件下载在这里\cv"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the template
template = DocxTemplate(TEMPLATE_PATH)

# Load the Excel file
df = pd.read_excel(EXCEL_PATH)

# Loop through each row in the Excel file
for index, row in df.iterrows():
    university_name = row['University Names']
    for col in ['Major1', 'Major2', 'Major3']:
        program = row[col]
        # Define context for the template
        context = {
            'University_Names': university_name,
            'Major': program
        }

        # Render the template with the context
        template.render(context)

        # Define output file names
        word_filename = f"{university_name.replace(' ', '_')}_{program.replace(' ', '_')}_Statement_of_Purpose.docx"
        pdf_filename = f"{university_name.replace(' ', '_')}_{program.replace(' ', '_')}_Statement_of_Purpose.pdf"

        # Save the Word document
        word_path = os.path.join(OUTPUT_DIR, word_filename)
        template.save(word_path)

        # Convert Word to PDF
        pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
        convert(word_path, pdf_path)

        print(f"Generated documents for {university_name} - {program}")

print("All documents have been generated successfully.")