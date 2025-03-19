#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
import pandas as pd
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import NoSuchElementException
import re


# In[2]:


driver = webdriver.Chrome()
url="https://esf.fang.com/house-a015277-b02655/"
driver.get(url)


# In[4]:


i=0
house_data=[]
cleaned_data=[]
while i<=3:
    houses=driver.find_elements(By.CSS_SELECTOR,"dl.clearfix")
    for house in houses:
        house_name=house.find_element(By.CSS_SELECTOR,".clearfix").text
        house_info=house.find_element(By.CSS_SELECTOR,".tel_shop").text
        house_price=house.find_element(By.CSS_SELECTOR,".price_right").text
        #transport=house.find_element(By.CSS_SELECTOR,".bg_none icon_dt").text
        item={"名字":house_name,"房子信息":house_info,"价格":house_price}
        house_data.append(item)
    try:
        next_page = driver.find_element(By.LINK_TEXT, "下一页")
        next_page.click()
    except NoSuchElementException:
        print("NoSuchElementException")
    i+=1
    


# In[77]:


def parse_price(price):
    price_match = re.search(r'(\d+)万', price)
    unit_price_match = re.search(r'(\d+)元/㎡', price)
    return {
        '总价(万)': int(price_match.group(1)) if price_match else None,
        '单价(元/㎡)': int(unit_price_match.group(1)) if unit_price_match else None
    }


# In[ ]:





# In[78]:


info=[]
for item in house_data:
    house_info=item["房子信息"].split("|")
    price_info=parse_price(item["价格"])
    cleaned_entry = { 
        '名字': item['名字'].replace('\n', ''),
        "房子布局":house_info[0],
        '面积(㎡)': float(house_info[1][:-2]),
            '楼层': house_info[2],
            '朝向':house_info[3] ,
            '建成年份':int(house_info[4][:-3]) ,
         **price_info
    }
    info.append(cleaned_entry)
df= pd.DataFrame(info)


# In[82]:


df.to_csv("/Users/macbookair/Documents/苏州街房子信息.csv",encoding="utf-8-sig")


# In[65]:





# In[66]:





# In[ ]:




