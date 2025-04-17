import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 线性回归
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.linear_model import LinearRegression

# 加载训练数据
train_df = pd.read_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.pkl')
train_df[train_df['价格'] < 16000000]
# 分割训练集特征与目标
X_train = train_df.drop(['Unnamed: 0','价格'], axis=1)
y_train = train_df['价格']

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 加载预测数据
pred_df = pd.read_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_test.pkl')

# 提取预测特征（注意排除ID和价格）
X_pred = pred_df.drop(['Unnamed: 0','ID'], axis=1)
# 调整order
X_pred = X_pred[X_train.columns]
print(set(X_train.columns) - set(X_pred.columns),'\n',set(X_pred.columns) - set(X_train.columns))
# 生成预测结果
predictions = model.predict(X_pred)

# 构建结果DataFrame
result_df = pd.DataFrame({
    'ID': pred_df['ID'],
    'Price': predictions
})

# 保存为CSV（保留4位小数）
result_df.to_csv('3-31predictions.csv', index=False, float_format='%.4f')

