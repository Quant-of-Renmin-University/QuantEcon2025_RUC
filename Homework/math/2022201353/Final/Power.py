import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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

# 2. 数据准备
X = data.drop(columns='price')
y = data['price']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
y_log = np.log(y).astype(np.float32)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_log.values.reshape(-1, 1)).flatten()


# ==== 数据划分使用对数标准化后的y ====
X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
    X_scaled, y, test_size=0.2, random_state=111)
X_test_scaled = scaler.transform(X_test)  # 注意：使用训练集的scaler

models_config = {
     'XGBoost': {
        'model': XGBRegressor(random_state=1212, tree_method='hist'),
        'params': {
            'n_estimators': [8888],
            'max_depth': [18],
            'learning_rate': [ 0.0013253409745846666],
            'reg_alpha': [0.04452668245418041],
            'reg_lambda': [0.00021407701267691252],
            'gamma': [0.07125765992004345],
            'min_child_weight': [5],
            'subsample': [0.5730871362635312],
            'colsample_bytree': [0.6799523229221603]
        }
    },
}

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

scoring = {
    'MAE': make_scorer(inverse_mae),
    'RMSE': make_scorer(inverse_rmse)
}

# ==== 修改部分：模型训练和评估流程 ====
performance_report = []
results = {}

for model_name in models_config:
    print(f"\n=== 训练 {model_name} ===")
    config = models_config[model_name]
    metrics = {'Model': model_name}

    gscv = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=6,
        scoring=scoring,
        refit='RMSE',
        return_train_score=True,
        n_jobs=-1
        )
    gscv.fit(X_scaled, y)

    # 获取最佳模型
    best_model = gscv.best_estimator_
    results[model_name] = {
        'model': gscv.best_estimator_,
        'params': gscv.best_params_
    }
    print(f"{model_name}训练完成，最佳参数: {gscv.best_params_}")

    feature_names = X.columns.tolist()

    # 获取特征重要性/系数
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = best_model.coef_.flatten()
    else:
        print(f"{model_name} 模型不支持特征重要性/系数提取")
        continue

    # 创建并保存特征重要性表格
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    filename = f'feature_importance_{model_name}.csv'
    feature_importance.to_csv(filename, index=False)
    print(f"特征重要性已保存到 {filename}\n")

    # 样本内评估
    y_train_pred_scaled = best_model.predict(X_scaled)
    metrics['In-sample MAE'] = inverse_mae(y, y_train_pred_scaled)
    metrics['In-sample RMSE'] = inverse_rmse(y, y_train_pred_scaled)

    # 交叉验证结果
    metrics['CV MAE'] = gscv.cv_results_['mean_test_MAE'][gscv.best_index_]
    metrics['CV RMSE'] = gscv.cv_results_['mean_test_RMSE'][gscv.best_index_]

    # 验证集评估
    y_val_pred_scaled = best_model.predict(X_val)
    metrics['Out-sample MAE'] = inverse_mae(y_val_scaled, y_val_pred_scaled)
    metrics['Out-sample RMSE'] = inverse_rmse(y_val_scaled, y_val_pred_scaled)

performance_report.append(metrics)


# 5. 生成报告表格
# -------------------------------------------------
report_df = pd.DataFrame(performance_report)
columns_order = ['Model', 'In-sample MAE', 'In-sample RMSE',
                 'Out-sample MAE', 'Out-sample RMSE',
                 'CV MAE', 'CV RMSE']
report_df = report_df[columns_order]

print("\n=== 性能报告 ===")
print(report_df.to_markdown(index=False))

def calculate_and_save(model_name, predictions_scaled):
    """使用原始量级计算异常值"""
    predictions = inverse_transform(predictions_scaled)

    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    is_outlier = (predictions < lower_bound) | (predictions > upper_bound)
    outlier_count = sum(is_outlier)

    # 保存原始量级预测结果
    result_df = test[['ID']].copy()
    result_df['Price'] = np.round(predictions, 2)
    result_df.to_csv(f'predictions_{model_name}.csv', index=False)

    return {
        'model': model_name,
        'outlier_count': outlier_count,
        'outlier_ratio': f"{outlier_count / len(predictions):.2%}",
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(upper_bound, 2),
        'predictions': predictions
    }

# 预测测试集并保存结果
outlier_stats = []
for model_name in models_config:
    if model_name in results:
        pred_scaled = results[model_name]['model'].predict(X_test_scaled)
    else:
        model = models_config[model_name]['model'].fit(X_scaled, y)
        pred_scaled = model.predict(X_test_scaled)


    stats = calculate_and_save(model_name, pred_scaled)
    outlier_stats.append(stats)
    print(f"{model_name}异常值统计:")
    print(f"  异常值数量: {stats['outlier_count']}/{len(stats['predictions'])}")
    print(f"  异常值比例: {stats['outlier_ratio']}")
    print(f"  正常值范围: [{stats['lower_bound']}, {stats['upper_bound']}]")

# 生成异常值综合报告
outlier_report = pd.DataFrame(outlier_stats)[
    ['model', 'outlier_count', 'outlier_ratio', 'lower_bound', 'upper_bound']
]
outlier_report.to_csv('outliers_summary.csv', index=False)
print("\n各模型异常值统计已保存到 outliers_summary.csv")

