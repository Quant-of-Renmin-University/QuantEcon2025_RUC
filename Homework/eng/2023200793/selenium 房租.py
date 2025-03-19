#!/usr/bin/env python
# coding: utf-8

# In[92]:


from selenium import webdriver
import pandas as pd
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import NoSuchElementException
import re


# In[93]:


driver = webdriver.Chrome()
url="https://zu.fang.com/house-a015277-b02655/"
driver.get(url)


# In[95]:


driver


# In[94]:


items=driver.find_elements(By.CSS_SELECTOR, "dd.info.rel")
house_data=[]

i=0
while i<=4:
    for driver in items :
        title_element = driver.find_element(By.CSS_SELECTOR, ".title a")
        title = title_element.get_attribute("title")
        house_info = driver.find_element(By.CLASS_NAME, "font15.mt12.bold").text
        address=driver.find_element(By.CSS_SELECTOR, "p.gray6.mt12").text
        traffic=driver.find_element(By.CSS_SELECTOR,"span.note.subInfor").text
        price=driver.find_element(By.CSS_SELECTOR,"span.price").text
        item={"名字":title,"房子信息":house_info,"地址":address,"traffic":traffic,"价格":price}
        house_data.append(item)
    try:
        next_page = driver.find_element(By.LINK_TEXT, "下一页")
        next_page.click()
    except NoSuchElementException:
        print("NoSuchElementException")
    i+=1



# In[88]:


cleaned_data=[]

for house in house_data:
    house_info=house["房子信息"].split("|")
    
    house={
    "名字":house["名字"],
    "出租情况":house_info[0],
    "房子布局":house_info[1],
    '面积(㎡)': house_info[2][:-1],
    "朝向":house_info[3] ,
     "地址":house['地址'],
     "交通":house['traffic'],
    "价格":int(house["价格"])}
    cleaned_data.append(house)
    
    

df= pd.DataFrame(cleaned_data)    


# In[90]:


df.to_csv("/Users/macbookair/Documents/苏州街租金信息.csv",encoding="utf-8-sig")


# In[81]:





# In[89]:


df


# In[ ]:




