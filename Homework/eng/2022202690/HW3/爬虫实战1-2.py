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
    houses = driver.find_elements(By.CSS_SELECTOR, 'dl.list.hiddenMap.rel')

    for house in houses:
        try:
            title = house.find_element(
                By.CSS_SELECTOR, 'p.title a').get_attribute('title')
            price = house.find_element(By.CSS_SELECTOR, 'span.price').text
            house_info = house.find_element(
                By.CSS_SELECTOR, 'p.font15.mt12.bold').text
            info_parts = [x.strip() for x in house_info.split('|')]
            area = house.find_element(
                By.CSS_SELECTOR, 'p.gray6.mt12').text.replace('\n', ' ')
            try:
                metro = house.find_element(
                    By.CSS_SELECTOR, 'span.note.subInfor').text
            except:
                metro = '无'

            item = {
                '标题': title,
                '价格(元/月)': price,
                '租赁类型': info_parts[0],
                '户型': info_parts[1],
                '面积': info_parts[2],
                '朝向': info_parts[3],
                '区域': area,
                '交通信息': metro
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
    url = 'https://lf.zu.fang.com/house-a010278/'
    filename = 'haidian-wanliu.csv'
    first_page = True

    driver.get(url)
    sleep(20)

    page_data = scrape_page_data(driver)
    save_data(page_data, filename, is_first_page=first_page)
    print("第1页数据处理完成")

    for i in range(19):
        next_page(driver)
        sleep(3)
        page_data = scrape_page_data(driver)
        save_data(page_data, filename)
        print(f"第{i+2}页数据处理完成")
    print("前20页处理完成")
    driver.quit()


if __name__ == '__main__':
    main()