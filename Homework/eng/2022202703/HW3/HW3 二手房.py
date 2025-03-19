from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd

# 初始化 Edge 浏览器驱动
driver = webdriver.Edge()

# 打开网页
url = 'https://esf.fang.com/house-a015277-b022/i31/'
driver.get(url)

data_list = []
page_count = 20

for _ in range(page_count):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'dd.price_right'))
        )

        price_elements = driver.find_elements(By.CSS_SELECTOR, 'dd.price_right')

        for price_element in price_elements:
            try:
                total_price_element = price_element.find_element(By.CSS_SELECTOR, 'span.red b')
                total_price = total_price_element.text
                unit_price_element = price_element.find_element(By.XPATH, './span[2]')
                unit_price = unit_price_element.text.replace('元/㎡', '')

                data_list.append({
                    '总价': total_price,
                    '单价': unit_price
                })
            except:
                print('某个房源信息获取失败')

        try:
            # 修改翻页定位
            next_page_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'p.last a'))
            )
            next_page_button.click()
            time.sleep(5)
        except:
            print("未找到翻页元素，或已到达最后一页")
            break

    except Exception as e:
        print(f"发生错误: {e}")
        break

df = pd.DataFrame(data_list)
excel_path = r"C:\桌面文件下载在这里\HW3\data.xlsx"
df.to_excel(excel_path, index=False)
print(df)
driver.quit()