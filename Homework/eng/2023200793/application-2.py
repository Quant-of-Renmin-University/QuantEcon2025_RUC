#!/usr/bin/env python
# coding: utf-8

# In[10]:


from docxtpl import DocxTemplate
import pypandoc


# In[11]:


programs=["MA in Econnomics","MA in Financial Enginnering","MA in Data Science"]
universities=[]
with open("/Users/macbookair/Documents/courses.csv","r") as file:
    content=file.readlines()
for line in content:
    universities.append(line.strip())


# In[12]:


doc = DocxTemplate("/Users/macbookair/Documents/template.docx")
#转化为template
for university in universities:
    for program in programs:
        content={"program":program,"university":university}
        doc.render(content)
        doc.save(f"/Users/macbookair/Documents/{university}_{program}.docx")


# In[13]:


def docx_convert_pdf(file_path,pdf_path):
    output = pypandoc.convert_file(file_path, 'pdf', outputfile=pdf_path)


# In[14]:


file_path=[]
pdf_path=[]
for university in universities:
    for program in programs:
        file_path.append(f"/Users/macbookair/Documents/{university}_{program}.docx")
        pdf_path.append(f"/Users/macbookair/Documents/{university}_{program}.pdf")


        


# In[41]:


for i in range(len(file_path)):
    docx_convert_pdf(file_path[i],pdf_path[i])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




