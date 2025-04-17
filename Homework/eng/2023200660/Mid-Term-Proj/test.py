import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
# 线性回归
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Load the data
file = '/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.csv'
pkl_file = '/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.pkl'
if not os.path.exists(pkl_file):
    df = pd.read_csv(file)

    df['建筑面积'] = df['建筑面积'].apply(lambda x: x.replace('㎡','')).astype(float)
    # 然后再填充缺失值，用均值填充
    df['建筑面积'].fillna(df['建筑面积'].mean(), inplace=True)
    df['east'] = df['房屋朝向'].apply(lambda x: 1 if '东' in x else 0)
    df['west'] = df['房屋朝向'].apply(lambda x: 1 if '西' in x else 0)
    df['south'] = df['房屋朝向'].apply(lambda x: 1 if '南' in x else 0)
    df['north'] = df['房屋朝向'].apply(lambda x: 1 if '北' in x else 0)
    df.drop('房屋朝向', axis=1, inplace=True)

    text_cols = [
        '核心卖点',
        '户型介绍',
        '周边配套',
        '交通出行'
    ]
    df.drop(text_cols, axis=1, inplace=True)

    df['lat2central']=df['lat'].apply(lambda x: (x-df['lat'].mean())**2)
    df['lon2central']=df['lon'].apply(lambda x: (x-df['lon'].mean())**2)
    df.drop(['lat','lon'],axis=1,inplace=True)
    
    chinese2num_dict = {
    '一': 1,
    '二': 2,
    '两': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9,
    '十': 10,
    '十一': 11,
    '十二': 12,
    '十三': 13,
    '十四': 14,
    '十五': 15,
    '十六': 16,
    '十七': 17,
    '十八': 18,
    '十九': 19,
    '二十': 20,
    '二十一': 21,
    '二十二': 22,
    '二十三': 23,
    '二十四': 24,
    '二十五': 25,
    '二十六': 26,
    '二十七': 27,
    '二十八': 28,
    '二十九': 29,
    '三十': 30,
    '三十一': 31,
    '三十二': 32,
    '三十三': 33,
    '三十四': 34,
    '三十五': 35,
    '三十六': 36,
    '三十七': 37,
    '三十八': 38,
    '三十九': 39,
    '四十': 40,
    '四十一': 41,
    '四十二': 42,
    '四十三': 43,
    '四十四': 44,
    '四十五': 45,
    '四十六': 46,
    '四十七': 47,
    '四十九': 49,
    '五十': 50,
    '五十一': 51,
    '五十二': 52,
    '五十三': 53,
    '五十四': 54,
    '五十五': 55,
    '五十六': 56,
    '五十七': 57,
    '五十八': 58,
    '五十九': 59,
    '六十': 60,
    '六十一': 61,
    '六十二': 62,
    '六十三': 63,
    '六十四': 64,
    '六十五': 65,
    '六十六': 66,
    '六十七': 67,
    '六十八': 68,
    '六十九': 69,
    '七十': 70,
    '七十一': 71,
    '七十二': 72,
    '七十三': 73,
    '七十四': 74,
    '七十五': 75,
    '七十六': 76,
    '七十七': 77,
    '七十八': 78,
    '七十九': 79,
    '八十': 80,
    '八十一': 81,
    '八十二': 82,
    '八十三': 83,
    '八十四': 84,
    '八十五': 85,
    '八十六': 86,
    '八十七': 87,
    '八十八': 88,
    '八十九': 89,
    '九十': 90,
    '九十一': 91,
    '九十二': 92,
    '九十三': 93,
    '九十四': 94,
    '九十五': 95,
    '九十六': 96,
    '九十七': 97,
    '九十八': 98,
    '九十九': 99,
    '一百': 100
    }
    df['lift_ratio'] = df['梯户比例'].apply(lambda x:
                                     chinese2num_dict[x.split('梯')[0]]/chinese2num_dict[x.split('梯')[1].replace('户','')]
                                     if '梯' in str(x) else 0)
    # 保存到pkl文件
    df.to_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.pkl')


    time_cols = [
        '交易时间','年份','上次交易'
    ]
    df.drop(time_cols, axis=1, inplace=True)
    


    first_one_hot_col = [
        '城市','环线','建筑结构','装修情况','别墅类型','配备电梯','房屋用途','房屋年限','产权所属'
    ]
    for col in first_one_hot_col:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
        df.drop(col, axis=1, inplace=True)
    # 保存到pkl文件

    first_drop_col = [
        '区域',
        '板块',
        '套内面积',
        '所在楼层',
        '小区名称',
        '房屋户型',
    ]
    df.drop(first_drop_col, axis=1, inplace=True)
    df.to_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.pkl')

    second_drop = ['梯户比例'
    ]
    df.to_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.pkl')

    print(df.columns)


    first_drop_col = [
        '交易权属',
        '房屋优势'
    ]
    df.drop(first_drop_col, axis=1, inplace=True)
    df.to_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.pkl')
    for col in df.columns:
        print(col.ljust(20), df[col].dtype)
else:
    df = pd.read_pickle(pkl_file)
    X = df.drop('价格', axis=1)
    y = df['价格']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    # 打印mse、rmse、mae
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)/len(y_test)
    rmse = np.sqrt(mse)/len(y_test)
    mae = mean_absolute_error(y_test, y_pred)/len(y_test)
    print('mse:', mse)
    print('rmse:', rmse)
    print('mae:', mae)
