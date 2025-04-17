## HW3: Selenium&DataAnalysis Score Report

Your unique report has been sent to your own email (e.g. 202x20xxx@ruc.edu.cn or xxxxxxxx@qq.com).



## Emphasis:

  - **!!Original work!!:** Plagiarism will not be tolerated.

- **Code Guide:** Refer to the Google Python Style Guide for elegant code writing

  - Chinese version: https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules.html
  - English version: https://github.com/google/styleguide
  - Sargent Code: https://github.com/Quant-of-Renmin-University/QuantEcon2025_RUC/blob/main/Codes/A_QuantEcon_Sargent/15_writing_good_code.ipynb

## Appendix:

#### ***Appendix 1:***

*MLR.3: No Perfect Collinearity.

The assumption only requires that the independent variables should not have a perfect linear relationship. For example, variables with a relationship like: $x_1 + x_2 = x_3$.

Therefore, we need to remove one category (as the base category) in the regression.

[1] Wooldridge, Jeffrey M. *Introductory Econometrics: A Modern Approach* 6th ed. Cengage Learning, 2016.

<a id="appendix2"></a>

#### ***Appendix 2:***

In regression analysis, categorical variables need proper encoding for correct model inclusion. Directly assigning values like 1, 2, 3, 4 to a categorical variable can mislead the model into treating it as a numerical variable with order and interval relationships, leading to incorrect interpretations.

To avoid this issue, use methods like:

***OneHotEncoder***: Create binary dummy variables (0 or 1) for each category.
<a id="appendix3"></a>

#### ***Appendix 3:***

In order to generate images in python that can be displayed in Chinese, we need to add codes:

```Python
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
```



