from selenium import webdriver
import pandas as pd
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


def go_to_next_page(browser):
    try:
        next_button = browser.find_element(By.PARTIAL_LINK_TEXT, "下一页")
        next_button.click()
        return True
    except NoSuchElementException:
        print("已到达最后一页，无法继续翻页")
        return False


def extract_house_info(browser):
    results = []
    listings = browser.find_elements(By.CLASS_NAME, 'clearfix')

    for listing in listings:
        try:
            title = listing.find_element(By.CSS_SELECTOR, 'h4.clearfix a').get_attribute('title')
            price_info = listing.find_element(By.CLASS_NAME, 'price_right').text.split()
            total_price = price_info[0]
            unit_price = price_info[1]
            details = listing.find_element(By.CLASS_NAME, 'tel_shop').text.split('|')
            details = [item.strip() for item in details]
            location_info = listing.find_element(By.CLASS_NAME, 'add_shop').text.split()
            community_name = location_info[0]
            district = location_info[1]
            try:
                metro_info = listing.find_element(By.CSS_SELECTOR, 'span.bg_none.icon_dt').text
            except NoSuchElementException:
                metro_info = '无'

            entry = {
                'title': title,
                'community': community_name,
                'total_price': total_price,
                'unit_price': unit_price,
                'layout': details[0],
                'size': details[1],
                'floor': details[2],
                'direction': details[3],
                'year': details[4],
                'district': district,
                'metro': metro_info
            }

            results.append(entry)
        except Exception as error:
            print(f"解析房源信息时出错: {error}")
            continue
    return results


def store_data(data, file_name, is_first=False):
    df = pd.DataFrame(data)
    df.to_csv(file_name, mode='a' if not is_first else 'w', header=is_first, index=False, encoding='utf-8-sig')


def initialize_browser():
    browser = webdriver.Edge()
    return browser


def scrape_and_save_data(browser, url, output_file):
    browser.get(url)
    time.sleep(20)

    is_first_page = True
    for page_num in range(1, 21):
        page_data = extract_house_info(browser)
        store_data(page_data, output_file, is_first_page)
        print(f"第{page_num}页数据已保存")
        is_first_page = False

        if not go_to_next_page(browser):
            break
        time.sleep(3)


def close_browser(browser):
    browser.quit()
    print("浏览器已关闭")


# 直接调用函数实现功能
browser = initialize_browser()
target_url = 'https://tj.esf.fang.com/house-a041-b0967/'
output_file = 'tianjin_balitai_housing_prices.csv'

scrape_and_save_data(browser, target_url, output_file)
close_browser(browser)