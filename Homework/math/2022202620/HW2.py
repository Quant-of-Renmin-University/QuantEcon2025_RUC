import os
import pandas as pd
from docx import Document
import docx2pdf
import time
import subprocess

# 生成申请信模板的函数
def generate_sop_template(name, university, department, program):
    sop = f"""
    Dear admission committee,

    My name is {name}, and I am writing to express my sincere interest in applying for admission to the {department} at {university}. 
    {university}'s reputation for academic excellence, particularly in the field of economics, aligns perfectly with my academic aspirations and career goals.

    Over the past few years, I have developed a strong foundation in economics, statistics, and data science, which I believe makes me well-prepared for the rigorous academic environment at {university}.
    I am particularly interested in pursuing the {program} at your esteemed institution. I am confident that this program will equip me with the necessary skills to contribute meaningfully to the field of economics and make significant strides in the industry.

    I have always been passionate about analyzing economic data and understanding the deeper mechanisms that govern the economy. My previous academic experiences have allowed me to build a strong quantitative background, and I am eager to expand my knowledge further under the guidance of your distinguished faculty.

    Thank you for considering my application. I look forward to the opportunity to contribute to and learn from the vibrant academic community at {university}.

    Sincerely,
    {name}
    """
    return sop

# 将申请信保存为 Word 文件的函数
def save_sop_as_word(sop, filename):
    doc = Document()
    doc.add_paragraph(sop)
    doc.save(filename)


# Excel 文件路径（请根据你的实际情况修改路径）
excel_path = 'C:/Users/86158/Desktop/HW2/Universities_Programs.xlsx'
df = pd.read_excel(excel_path)

# 确定Word和PDF文件夹路径
word_folder = r"C:\Users\86158\Desktop\HW2\output\Word"
pdf_folder = r"C:\Users\86158\Desktop\HW2\output\PDF"

# 创建文件夹（如果它们不存在）
os.makedirs(word_folder, exist_ok=True)
os.makedirs(pdf_folder, exist_ok=True)

# 生成所有Word文件
for index, row in df.iterrows():
    university = row['University Names']
    department = "Economics Department"
    programs = [row['Major1'].strip(), row['Major2'].strip(), row['Major3'].strip()]
    
    for i, program in enumerate(programs, start=1):
        name = "Chenglin Wang"  
        sop = generate_sop_template(name, university, department, program)
        
        filename = f"{word_folder}/StatementOfPurpose_{university.replace(' ', '_')}_{i}.docx"
        save_sop_as_word(sop, filename)
        print(f"Statement of Purpose for {program} at {university} has been saved as {filename}")

# 将所有Word文件转换为PDF
for filename in os.listdir(word_folder):
    if filename.endswith('.docx'):
        docx_path = os.path.join(word_folder, filename)
        pdf_path = os.path.join(pdf_folder, filename.replace('.docx', '.pdf'))
        docx2pdf.convert(docx_path, pdf_path)
        print(f"Converted {filename} to PDF as {pdf_path}")