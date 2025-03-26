| 学号       | 姓名     | 班级                 | group | HW3 | Tips |
|------------|-------------|----------------------|-------|-----|------|
| 2023200793 | 江智睿 | 2023经济学类1班      | 1     |8     |1. 二手房与租房爬取的`block`不一致。<br>2.没有按照要求画出预测的两个柱状图。 <br>3. 可以看出作者发现了回归系数异常的问题并采取了绘制相关系数图的方法删去共线变量，但归根结底问题在于`block`的哑变量，详见[Appendix1](#appendix1).    |
| 2021200697 | 柳任时 | 2021经济学3班       | 1     |7     |1. 翻页和爬数据可以写成循环，不必来回运行。<br>2. 没有数据处理和分析。      |
| 2022202560 | 卢俊然 | 2022经济学类4班      | 1     |8     |1. 北太平庄只有1页，但最好设置计数也应按要求中的`20`做，可以在循环中设置`break`在发现不能翻页时提早结束。<br>2. 正确使用`pd.get_dummies`提取了哑变量，但是线性回归中忽略了共线性问题，可以参考[Appendix1](#appendix1)。<br>3. 预测时`interactions`的值设置存在问题。<br> 4.注释过少，代码的重点部分没有注释。     |
| 2023200773 | 钱彦均 | 2023经济学类1班      | 2     |8     |1.代码太乱了，可以将代码清洁一下，例如出错的`检查翻页`部分就可以不放在代码里，提取不同的数据也无需重新导包。<br>2. 正确使用`pd.get_dummies`提取了哑变量，但是线性回归中忽略了共线性问题，可以参考[Appendix1](#appendix1)。     |
| 2022202703 | 陈相安 | 2022经济学类3班      | 2     |8.5     |1.回归的时候没有构建`interactions`，而且区域变量最好应使用独热编码而非序数，可以参考[Appendix2](#appendix2)。<br>2.绘制图像和分析比较出色，但图像没有支持中文显示，可以参考[Appendix3](#appendix3)。      |
| 2023200660 | 段毅   | 2023经济学类4班      | 2     |9.5     |1.设置了众多`ChromeOptions`，有利于网页爬取的稳健性。<br>2.数据爬取中采用了`tqdm`库实时查看进度条。<br>3.正确使用`pd.get_dummies`提取了哑变量，但是线性回归中忽略了共线性问题，可以参考[Appendix1](#appendix1)。      |
| 2023200584 | 张亦佳 | 2023经济学类4班      | 2     |8.5     |1.数据爬取时注意`if`和`elif`逻辑的区别。<br>2.清洗数据中保留数字就好了，不用刻意加单位。<br> 3. 区域变量最好采用独热编码而非序数，可以参考[Appendix1](#appendix1)。      |
| 2022202690 | 王煜阳 | 2022经济学类4班      | 3     |5     |      |
| 2022200230 | 妙含笑 | 2022级本科金融工程班  | 3     |9.5     |1. 函数逻辑严密，数据分析、线性回归和作图的细节都处理的很好，但两个爬取函数最好不应同名，可以试试集成在一个函数中。<br>2.后续绘图的数字已存储在`result`中，就不用手敲了。     |
| 2022202622 | 薛佳易 | 2022经济学类2班      | 3     |9     |1.整体来说没什么问题，但是不应把房价和租房信息`merge`（并不是一套房子），后续计算房价租金比率也会出问题。      |
| 2023200739 | 钟梓洋 | 2023经济学类3班      | 4     |8     |1.封装函数爬取网页的思路固然好，但翻页的功能未实现，并且函数内部缺少判断，运行并不稳健。     |
| 2023200579 | 王锦川 | 2023经济学类3班      | 4     |8     |1.`下一页`按钮找不到的话考虑重进链接是一个很好的思路，但是代码中并没有实现。      |
| 2021200567 | 赵一铭 | 2021经济学1班       | 5     |8     |1.租房部分`url`爬取的是整个天津市，但是`table`中只有滨海新区。 <br>2. 可以尝试使用`Python`清洗数据，例如：使用`re`函数提取数字，线性回归中生成哑变量可以使用`pd.get_dummies`。     |
| 2021200639 | 邱丹绮 | 2021国际经济与贸易1班| 5     |8.5     |1.提取函数很详细，可以考虑将爬取也封装成一个函数。 <br>2. 可以尝试使用`Python`清洗数据，例如：使用`re`函数提取数字，线性回归中生成哑变量可以使用`pd.get_dummies`。      |
| 2022202751 | 张雷臻 | 2022经济学类3班      | 5     |9     |1. 爬取也可以像清洗数据一样封装成函数。<br>2. 数据`merge`时没有必要把租房和二手房合并在一起，会产生歧义。 <br>3. 使用了`OneHotEncoder`提取哑变量，但忽略了共线性问题，可以参考[Appendix1](#appendix1)。      |
| 2022202777 | 晏子晗 | 2022经济学类1班      | 5     |8     |1. 数据爬取和清洗均封装为了函数，可以直接调用。 <br> 2. 面积为100m<sup>2</sup>的交互项构建的有问题。 <br>3. 两个预测的柱状图只画了一个，两个model建议用不同的名字。 <br>4. 忽略了共线性问题，可以参考[Appendix1](#appendix1)。     |
| 2021200641 | 林亦隆 | 2021经济学3班       | 6     |     |      |
| 2021200631 | 方至成 | 2021经济学-数学双学位实验班 | 6     |     |      |




  - **!!Original work!!:** Plagiarism will not be tolerated.

- **Code Guide:** Refer to the Google Python Style Guide for elegant code writing

  - Chinese version: https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules.html
  - English version: https://github.com/google/styleguide
  - Sargent Code: https://github.com/Quant-of-Renmin-University/QuantEcon2025_RUC/blob/main/Codes/A_QuantEcon_Sargent/15_writing_good_code.ipynb

## Appendix:
<a id="appendix1"></a>
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




