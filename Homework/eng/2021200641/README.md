```markdown

## 结构
```
project/
├── data_source.ipynb          # 数据提取主文件
├── processed_data/
│   └── merge_chars_60.ipynb   # 因子数据合并
│   └──impute_rank_output_bckmk.ipynb  # 因子值排序处理
├── factor_calculation/        # 各因子计算脚本
└── README.md
```

### 运行顺序
1. **数据提取**
   - 运行 `data_source.ipynb` 提取所有需要的数据
   - 可以尝试切换不同的WRDS数据库账号进行数据提取，避免速度限制

2. **因子计算**
   - 运行所有因子计算文件
   - 注意：部分脚本使用了multiprocessing(占用8-10核CPU)
   - 计算资源足够时可convert为.py文件并行运行

3. **数据合并**
   - 运行 `processed_data/merge_chars_60.ipynb` 合并所有因子数据

4. **排序处理**
   - 运行 `impute_rank_output_bckmk.ipynb` 得到因子值的排序数据
   - 注意：排序后损失了因子值分布信息，后续使用需谨慎

## 优化

### 计算优化
1. 回归方法改进：
   - 对beta、residual_variance等使用加权最小二乘代替OLS
   - 半衰期设置为1.5个月(rolling window为3个月)

2. Rolling计算优化：
   - 部分rolling window计算使用指数加权
   - 半衰期设置为1.5个月(rolling window为3个月)

3. 并行计算优化：
   - 根据month_num字段均匀分配子df大小
   - 提高multiprocessing的计算效率

4. 回归计算：
   - 使用statsmodel.api替代了显式回归计算
