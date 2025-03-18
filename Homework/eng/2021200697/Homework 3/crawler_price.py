from selenium import webdriver
import pandas as pd
from time import sleep
from selenium.webdriver.common.by import By
from selenium.common import NoSuchElementException


def next_page(driver):
    try:
        next_page = driver.find_element(By.LINK_TEXT, "下一页")
        next_page.click()
    except NoSuchElementException:
        print("NoSuchElementException")


def scrape_page_data(driver):
    data = []
    houses = driver.find_elements(By.CSS_SELECTOR, 'dl.clearfix')

    for house in houses:
        try:
            title = house.find_element(
                By.CSS_SELECTOR, "h4.clearfix a[title]"
            ).get_attribute("title")
            house_info = house.find_element(
                By.CSS_SELECTOR, 'p.tel_shop').text
            info_parts = [x.strip()
                          for x in house_info.split("|") if x.strip()]
            community = house.find_element(
                By.CSS_SELECTOR, 'p.add_shop a').get_attribute('title')
            address = house.find_element(
                By.CSS_SELECTOR, 'p.add_shop span').text
            price_dd = house.find_element(By.CSS_SELECTOR, "dd.price_right")
            total_price = price_dd.find_element(
                By.CSS_SELECTOR, "b").text.strip()
            unit_price = price_dd.find_elements(
                By.CSS_SELECTOR, "span")[-1].text.strip()

            item = {
                '标题': title,
                '户型': info_parts[0],
                '面积': info_parts[1],
                '层数': info_parts[2],
                '朝向': info_parts[3],
                '建成时间': info_parts[4],
                '小区': community,
                '具体地址': address,
                '总价（万）': total_price,
                '单价': unit_price
            }
            data.append(item)
        except Exception as e:
            print(f'解析条目时出错：{str(e)}')
            continue
    return data


def save_data(data, filename, is_first_page=False):
    df = pd.DataFrame(data)
    df.to_csv(filename,
              mode='a' if not is_first_page else 'w',
              header=is_first_page,
              index=False,
              encoding='utf_8_sig')


def main():
    driver = webdriver.Edge()
    url = 'https://esf.fang.com/house-a015277-b02313/'
    filename = 'haidian-wanliu-price.csv'
    first_page = True

    driver.get(url)
    sleep(10)

    page_data = scrape_page_data(driver)
    save_data(page_data, filename, is_first_page=first_page)
    print("第1页数据处理完成")

    for i in range(19):
        next_page(driver)
        sleep(7)
        page_data = scrape_page_data(driver)
        save_data(page_data, filename)
        print(f"第{i+2}页数据处理完成")
    print("前20页处理完成")
    driver.quit()


if __name__ == '__main__':
    main()
