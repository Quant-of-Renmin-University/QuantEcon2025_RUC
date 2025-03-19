from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd

# 初始化 Edge 浏览器驱动
driver = webdriver.Edge()

# 打开网页
url = 'https://zu.fang.com/house-a015277-b022/'
driver.get(url)

data_list = []
page_count = 6  # 设置翻页次数

try:
    for _ in range(page_count):
        # 等待页面加载完成
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'p.font15.mt12.bold'))
        )

        # 获取所有匹配的元素
        elements = driver.find_elements(By.CSS_SELECTOR, 'p.font15.mt12.bold')

        for element in elements:
            # 获取元素的文本内容
            text = element.text

            # 使用分隔符分割文本
            parts = text.split('|')

            # 去除空格和换行符
            cleaned_parts = [part.strip() for part in parts]

            # 定位当前房子的月租价格
            try:
                parent = element.find_element(By.XPATH, 'ancestor::dd')
                price_element = parent.find_element(By.CSS_SELECTOR, 'div.moreInfo p.mt5.alingC span.price')
                monthly_rent = price_element.text
            except:
                monthly_rent = "价格未找到"

            # 将数据添加到列表
            cleaned_parts.append(monthly_rent)
            data_list.append(cleaned_parts)

        # 尝试翻页
        try:
            # 使用 XPath 定位“下一页”按钮
            next_page_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='fanye']/a[text()='下一页']"))
            )
            next_page_button.click()
            time.sleep(3)  # 等待页面加载
        except:
            print("未找到翻页元素，或已到达最后一页")
            break

except Exception as e:
    print(f"发生错误: {e}")

# 打印结果
for data in data_list:
    print(data)

df = pd.DataFrame(data_list)
excel_path = r"C:\桌面文件下载在这里\HW3\data2.xlsx"
df.to_excel(excel_path, index=False)

# 关闭浏览器驱动
driver.quit()