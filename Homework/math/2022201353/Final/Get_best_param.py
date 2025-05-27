import pandas as pd
import optuna
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os
from sklearn.model_selection import KFold
os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8'



# 1. 数据读取
data = pd.read_parquet('train_features_target.parquet')
X_test = pd.read_parquet('test_features_target.parquet')
test = pd.read_csv('ruc_Class25Q1_test.csv')
oof_train1 = pd.read_parquet('train_oof_predictions1.parquet')
oof_test1 = pd.read_parquet('test_oof_predictions1.parquet')
oof_train2 = pd.read_parquet('train_oof_predictions2.parquet')
oof_test2 = pd.read_parquet('test_oof_predictions2.parquet')
oof_train_second = pd.read_parquet('train_oof_predictions_second.parquet')
oof_test_second = pd.read_parquet('test_oof_predictions_second.parquet')
oof_train_ANN = pd.read_parquet('train_oof_predictions_ANN.parquet')
oof_test_ANN = pd.read_parquet('test_oof_predictions_ANN.parquet')
data = pd.concat([data, oof_train1], axis=1)
#data = pd.concat([data, oof_train2], axis=1)
data = pd.concat([data, oof_train_second], axis=1)
data = pd.concat([data, oof_train_ANN], axis=1)
X_test = pd.concat([X_test, oof_test1], axis=1)
#X_test = pd.concat([X_test, oof_test2], axis=1)
X_test = pd.concat([X_test, oof_test_second], axis=1)
X_test = pd.concat([X_test, oof_test_ANN], axis=1)
X_test = X_test.drop(columns='ID')
print(X_test.columns)
# 2. 数据准备
X = data.drop(columns='price')
y = data['price'].values #numpy数组

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
y_log = np.log(y).astype(np.float32)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1)).flatten().astype(np.float32)

# 4. 数据划分（保持为NumPy数组）
X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=111
)

# 5. 测试集处理（保持为NumPy数组）
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# 模型训练与评估
# -------------------------------------------------
def inverse_transform(y_scaled):
    """逆变换步骤：标准化逆变换 -> 指数变换"""
    y_log = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    return np.exp(y_log)
def inverse_mae(y_true_scaled, y_pred_scaled):
    y_true = inverse_transform(y_true_scaled)
    y_pred = inverse_transform(y_pred_scaled)
    return mean_absolute_error(y_true, y_pred)
def inverse_rmse(y_true_scaled, y_pred_scaled):
    y_true = inverse_transform(y_true_scaled)
    y_pred = inverse_transform(y_pred_scaled)
    return np.sqrt(mean_squared_error(y_true, y_pred))



# 模型配置
models_config = {
    'XGBoost': {
        'model': XGBRegressor(
            random_state=1212,
            tree_method='hist',
            device='cuda:0',
            n_estimators=5000,
            enable_categorical=True,
        ),
        'optuna_params': {
            'max_depth': (45, 50),
            'learning_rate': (0.001, 0.1),
            'reg_alpha': (0.04, 0.6),
            'reg_lambda': (0.0001, 0.0005),
            'gamma': (0.0001, 0.1),
            'min_child_weight': (5, 8),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1)
        }
    }
}


def objective(trial, model_name, X, y):
    config = models_config[model_name]

    params = {
        'max_depth': trial.suggest_int('max_depth', *config['optuna_params']['max_depth']),
        'learning_rate': trial.suggest_float('learning_rate', *config['optuna_params']['learning_rate'], log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', *config['optuna_params']['reg_alpha'], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', *config['optuna_params']['reg_lambda'], log=True),
        'gamma': trial.suggest_float('gamma', *config['optuna_params']['gamma']),
        'min_child_weight': trial.suggest_int('min_child_weight', *config['optuna_params']['min_child_weight']),
        'subsample': trial.suggest_float('subsample', *config['optuna_params']['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *config['optuna_params']['colsample_bytree']),
        'tree_method': 'hist',
        'device': 'cuda:0'
    }

    # 交叉验证配置
    n_splits = 5  # 可调整折数或从配置读取
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]


        model = config['model'].set_params(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae = inverse_mae(y_val, preds)
        rmse = inverse_rmse(y_val, preds)

        mae_scores.append(mae)
        rmse_scores.append(rmse)

    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)

    return -((0.9 * avg_mae + 0.1 * avg_rmse) / 2400) + 100



scoring = {
    'MAE': make_scorer(inverse_mae),
    'RMSE': make_scorer(inverse_rmse)
}


performance_report = []
results = {}

for model_name in models_config:
    print(f"\n=== Optimizing {model_name} with Optuna ===")
    config = models_config[model_name]
    metrics = {'Model': model_name}

    # Optuna优化
    study = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(
        lambda trial: objective(trial, model_name, X_scaled, y_scaled),
        n_trials=100,
        timeout=18000
    )

    # 获取最佳参数
    best_params = study.best_params
    print(f"Best params for {model_name}:")
    print(study.best_params)


