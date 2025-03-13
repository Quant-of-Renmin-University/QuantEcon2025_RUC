
import pandas as pd
from docx2pdf import convert  # Windows用户取消注释这行

# 读取Excel数据
df = pd.read_excel("university_data.xlsx")

# 循环处理每一行
for index, row in df.iterrows():
    university = row['University Name']
    programs = [row["Major 1"], row["Major 2"], row["Major 3"]]

    # 为每个专业生成文件
    for program in programs:
        # 填充模板
        doc = DocxTemplate("template.docx")
        context = {
            "university": university,
            "program": program
        }
        doc.render(context)

        # 保存Word文件
        filename = f"{university}_{program.replace(' ', '_')}"
        doc.save(f"output/{filename}.docx")

        # 转换为PDF（仅Windows）
        # convert(f"output/{filename}.docx")  # Windows用户取消注释这行

print("所有文件已生成！")