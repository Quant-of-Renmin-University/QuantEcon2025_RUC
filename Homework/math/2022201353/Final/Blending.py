import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import os
os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8'

# 1. 数据读取
data = pd.read_parquet('train_features_target.parquet')
X_test = pd.read_parquet('test_features_target.parquet')
test = pd.read_csv('ruc_Class25Q1_test.csv')
oof_train1 = pd.read_parquet('train_oof_predictions1.parquet')
oof_test1 = pd.read_parquet('test_oof_predictions1.parquet')
oof_train2 = pd.read_parquet('train_oof_predictions2.parquet')
oof_test2 = pd.read_parquet('test_oof_predictions2.parquet')
data = pd.concat([data, oof_train1], axis=1)
data = pd.concat([data, oof_train2], axis=1)
X_test = pd.concat([X_test, oof_test1], axis=1)
X_test = pd.concat([X_test, oof_test2], axis=1)
X_test = X_test.drop(columns='ID')


# 2. 数据准备
X = data.drop(columns='price')
y = np.log(data['price'])  # 仅对数变换

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. 基模型配置
base_models = [
    ('second Ridge', Ridge(alpha=2222, random_state=42)),
    ('second XGBoost', XGBRegressor(
        n_estimators=9999,
        max_depth=16,
        learning_rate=0.031270560944216645,
        reg_alpha=0.0441812975251821,
        reg_lambda=0.00012889288851753892,
        gamma=0.009043817632267573,
        min_child_weight=5,
        subsample=0.6397621471255355,
        colsample_bytree=0.6491364351554741,
        random_state=42,
        tree_method='hist',
    )),
    ('second MLP', MLPRegressor(
        random_state=42,
        hidden_layer_sizes=(389, 137),  # 网络结构
        activation='relu',  # 激活函数
        alpha=0.00037011770791260093,  # L2正则化强度
        learning_rate_init=0.0009355731198851312,  # 初始学习率
        batch_size=72,  # 批次大小
        solver='adam'  # 优化器
    ))

]

# 4. 生成OOF预测
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_train = np.zeros((X_scaled.shape[0], len(base_models)))  # 训练集的OOF预测
oof_test = np.zeros((X_test_scaled.shape[0], len(base_models)))  # 测试集的OOF预测

# 遍历每个基模型
# 在交叉验证循环内部处理标准化
for model_idx, (name, model) in enumerate(base_models):
    print(f"\n=== 处理模型 {name} ===")
    test_preds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        # 划分原始数据（未标准化）
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 仅在训练部分拟合scaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        # 训练模型并预测验证集
        model.fit(X_train, y_train)
        oof_train[val_idx, model_idx] = model.predict(X_val)

        # 用当前fold的scaler转换测试集并预测
        X_test_scaled = scaler.transform(X_test)
        test_preds.append(model.predict(X_test_scaled))

    # 测试集预测取各fold平均
    oof_test[:, model_idx] = np.mean(test_preds, axis=0)

# 保存训练集的OOF预测
oof_train_df = pd.DataFrame(
    oof_train,
    columns=[name for name, _ in base_models],  # 使用基模型名称作为列名
    index=data.index                             # 保留原始数据的索引
)
oof_train_df.to_parquet('train_oof_predictions.parquet')

# 保存测试集的OOF预测
oof_test_df = pd.DataFrame(
    oof_test,
    columns=[name for name, _ in base_models]    # 使用基模型名称作为列名
)
oof_test_df['ID'] = test['ID'].values            # 添加测试集ID列
oof_test_df.to_parquet('test_oof_predictions.parquet', index=False)

# 5. 元模型训练
meta_model = LinearRegression()
print(f"\n=== 处理元模型 ===")
meta_model.fit(oof_train, y)  # 使用全部训练数据

# 6. 完整训练流程
final_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1
)
final_model.fit(X_scaled, y)  # 全量数据训练

# 7. 验证集评估
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 生成验证集预测
val_preds = np.zeros((X_val.shape[0], len(base_models)))
for model_idx, (name, model) in enumerate(base_models):
    model.fit(X_train, y_train)  # 仅用训练部分
    val_preds[:, model_idx] = model.predict(X_val)
val_stacking = meta_model.predict(val_preds)


# 8. 评估函数
def inverse_eval(y_true_log, pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(pred_log)
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred))
    )


val_mae, val_rmse = inverse_eval(y_val, val_stacking)
print(f"\n验证集评估：MAE={val_mae:.2f}  RMSE={val_rmse:.2f}")
print("模型系数（权重）:")
for i, coef in enumerate(meta_model.coef_):
    print(f"coef for 基模型{i}: {coef:.4f}")  # 保留4位小数

# 9. 最终预测
test_stacking = meta_model.predict(oof_test)
final_pred = np.exp(test_stacking)

# 10. 结果保存
test[['ID']].assign(Price=np.round(final_pred, 2)).to_csv('predictions_stacking_fixed.csv', index=False)

