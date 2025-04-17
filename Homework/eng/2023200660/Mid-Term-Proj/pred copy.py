import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 线性回归
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# 加载训练数据
train_df = pd.read_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_train.pkl')
train_df = train_df[train_df['价格'] < 16000000]   # 剔除价格异常值
# 分割训练集特征与目标
X_train = train_df.drop(['Unnamed: 0','价格'], axis=1)
y_train = train_df['价格']

pred_df = pd.read_pickle('/home/duanyi_prv/Project/SchoolCourse/QuantEcon2025_RUC/Homework/eng/2023200660/Mid-Term-Proj/ruc_Class25Q1_test.pkl')

# 提取预测特征（注意排除ID和价格）
X_pred = pred_df.drop(['Unnamed: 0','ID'], axis=1)
# 调整order
X_pred = X_pred[X_train.columns]


# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_pred = scaler.transform(X_pred)

y_train = y_train.values.reshape(-1, 1)

# 转换为 PyTorch Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_pred_tensor = torch.FloatTensor(X_pred)

# 定义神经网络
class ThreeLayerNet(nn.Module):
    def __init__(self, input_size):
        super(ThreeLayerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),   # 输入层 -> 隐藏层1
            nn.ReLU(),
            nn.Linear(32, 8),           # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Linear(8, 1)             # 隐藏层2 -> 输出层
        )
    
    def forward(self, x):
        return self.net(x)

# 初始化模型
model = ThreeLayerNet(input_size=len(X_train[0]))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs = 100
batch_size = 128

for epoch in range(epochs):
    # 随机打乱数据
    permutation = torch.randperm(X_train_tensor.size()[0])
    
    for i in range(0, X_train_tensor.size()[0], batch_size):
        # 获取小批量数据
        indices = permutation[i:i+batch_size]
        batch_X = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    predictions = model(X_pred_tensor).numpy().flatten()

# 保存结果
result_df = pd.DataFrame({
    'ID': pred_df['ID'],
    'Price': predictions
})
result_df.to_csv('./nn_predictions.csv', index=False, float_format='%.4f')