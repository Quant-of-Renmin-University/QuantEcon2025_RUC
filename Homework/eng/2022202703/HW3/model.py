import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 读取文件
excel_file = pd.ExcelFile('/mnt/model.xlsx')

# 获取二手房数据
house_sale = excel_file.parse('Sheet1')
# 重命名列名
house_sale.columns = ['二手房面积', '二手房每平米单价', '区域编码']

# 获取租房数据
house_rent = excel_file.parse('Sheet2')
# 重命名列名
house_rent.columns = ['面积', '总租金', '每平方米租金', '区域编码']

# 构建 Model 1 的自变量和因变量
X1 = sm.add_constant(house_sale[['二手房面积', '区域编码']])
y1 = house_sale['二手房每平米单价']

# 构建 Model 2 的自变量和因变量
X2 = sm.add_constant(house_rent[['面积', '区域编码']])
y2 = house_rent['每平方米租金']

# 拟合 Model 1
model1 = sm.OLS(y1, X1).fit()

# 拟合 Model 2
model2 = sm.OLS(y2, X2).fit()

# 准备预测数据
areas = [50, 100]
locations = house_sale['区域编码'].unique()

results = []
for location in locations:
    for area in areas:
        # 构建预测 Model 1 的数据
        X1_pred = pd.DataFrame({'const': [1], '二手房面积': [area], '区域编码': [location]})
        # 预测二手房每平米价格
        predicted_price = model1.predict(X1_pred)[0]

        # 构建预测 Model 2 的数据
        X2_pred = pd.DataFrame({'const': [1], '面积': [area], '区域编码': [location]})
        # 预测每平方米租金
        predicted_rent = model2.predict(X2_pred)[0]

        # 计算租售比
        price_rent_ratio = predicted_price / predicted_rent

        results.append({
            '区域编码': location,
            '面积': area,
            '二手房每平米价格': predicted_price,
            '每平方米租金': predicted_rent,
            '租售比': price_rent_ratio
        })

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 绘制不同区域不同面积下租售比的柱状图
pivot_df = results_df.pivot(index='面积', columns='区域编码', values='租售比')
ax = pivot_df.plot(kind='bar')

plt.xlabel('面积')
plt.ylabel('租售比')
plt.title('不同区域不同面积下的租售比')
plt.xticks(rotation=0)
plt.legend(title='区域编码')

plt.show()

print("不同区域不同面积下的预测结果：")
print(results_df)