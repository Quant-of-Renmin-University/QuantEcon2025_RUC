import pandas as pd
from docx import Document
from docx2pdf import convert


#1导入本地学校排名和模板的文件名

list_path="uni_list.xlsx"
input_path="template.docx"

#2定义函数

def readlist(input_file):
    
    #读取学校排名
    uni_list=pd.read_excel(input_file)
    return uni_list

def replace_docx(input_file, output_file, replacements):
    """
    替换 .docx 文件中的占位符。

    :param input_file: 输入的 .docx 文件路径
    :param output_file: 输出的 .docx 文件路径
    :param replacements: 一个字典，键为占位符，值为替换内容
    """
    # 加载 .docx 文件
    doc = Document(input_file)

    # 遍历所有段落
    for paragraph in doc.paragraphs:
        for placeholder, replacement in replacements.items():
            if placeholder in paragraph.text:
                # 替换占位符
                paragraph.text = paragraph.text.replace(placeholder, replacement)

    # 保存修改后的文件
    doc.save(output_file)

#3初始化

output_path="SOP_{}_{}.docx"
uni_list=readlist(list_path)
University="initial"
Major="initial"

#字典用于替换占位符
replacements={
    "{Major}": Major,
    "{University}": University
}

#4生成SOP
for i in range(30):
    #排名为i+1的学校
    University=uni_list.iloc[i,0]
    
    for j in range(3):
        #第j+1个专业
        Major=uni_list.iloc[i,j+1]

        #更新replacements字典
        replacements={
            "{Major}": Major,
            "{University}": University
        }

        #生成SOP且将它存到“SOP_University_Major.docx"文件中
        SOP_name=output_path.format(University,Major)
        replace_docx(input_path,SOP_name,replacements)

        #生成pdf文件
        convert(SOP_name,"SOP_{}_{}.pdf".format(University,Major))

#5运行完成
print("all done!")
        