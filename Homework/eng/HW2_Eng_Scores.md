| 学号       | 姓名     | 班级                 | group | HW1 | HW2   | Tips |
|------------|-------------|----------------------|-------|-----|-------|-----------------------|
| 2023200793 | 江智睿 | 2023经济学类1班      | 1     | 10  | 8    | 1.下次请交`ipynb`文件； 2.有一处注释，可以再多点； 3.文件路径合并可以使用`os`库    |
| 2021200697 | 柳任时 | 2021经济学3班       | 1     | 10  | 8     | 1.下次请交`ipynb`文件； 2.无注释； 3.`convert`文件夹不应在`for`循环中（选做不扣分）     |
| 2022202560 | 卢俊然 | 2022经济学类4班      | 1     | 10  | 8     | 1.无注释； 2.不用重复导包    |
| 2023200773 | 钱彦均 | 2023经济学类1班      | 2     | 10  | 9.5      | 1.代码清晰，有一定注释； 2.使用了两种方法实现任务； 3.运行过程中`print`实时查看进度   |
| 2022202703 | 陈相安 | 2022经济学类3班      | 2     | 10  | 8.5     | 1.下次请交`ipynb`文件；2.注释清晰； 3.应将`convert`置于`for`循环外，更有利于代码的稳健    |
| 2023200660 | 段毅   | 2023经济学类4班      | 2     | 10  | 9     | 1.有一定注释（代码前半部分无注释）；2.封装为了函数，调用起来更便捷； 3.先生成`doc`文件，再转为`pdf`，运行过程中`print`实时查看进度，代码稳健    |
| 2023200584 | 张亦佳 | 2023经济学类4班      | 2     | 10  | 8     | 1.无注释；2.试试调用`os`库，先设置好文件夹再生成相应文件； 3.`convert`置于`for`循环外会更稳健    |
| 2022202690 | 王煜阳 | 2022经济学类4班      | 3     | 10    | 晚交      |     |
| 2022200230 | 妙含笑 | 2022级本科金融工程班  | 3     | 10  | 8.5      | 1.导包最好全部放在开头； 2.有一定注释； 3. 先生成`doc`文件，再转为`pdf`，运行过程中`print`实时查看进度，代码稳健   |
| 2022202622 | 薛佳易 | 2022经济学类2班      | 3     | 10  | 8.5     | 1.不用重复导包； 2.有一定注释； 3.前面的`pip install`在正式提交时可以删除； 4.`convert`置于`for`循环外会更稳健   |
| 2023200739 | 钟梓洋 | 2023经济学类3班      | 4     | 10  | 9   | 1.下次请交`ipynb`文件；2.注释清晰； 3.循环和判断逻辑严密，可以在`if`语句后加上`else`使得代码更加稳健    |
| 2023200579 | 王锦川 | 2023经济学类3班      | 4     | 10  | 8     | 1.没有注释    |
| 2021200567 | 赵一铭 | 2021经济学1班       | 5     | 10  | 8     | 1.可以统筹安排每一个`cell`，例如导包可以放在一起； 2.没有注释； 3.可以试试使用`os`库统筹安排文件    |
| 2021200639 | 邱丹绮 | 2021国际经济与贸易1班| 5     | 10  | 9     | 1.注释清晰； 2. 运行过程中`print`实时查看进度   |
| 2022202751 | 张雷臻 | 2022经济学类3班      | 5     | 10  | 8.5   | 1.注释清晰； 2.不同的步骤建议在不同的`cell`中运行； 3.代码报错因为未将`convert`置于`for`循环外    |
| 2022202777 | 晏子晗 | 2022经济学类1班      | 5     | 10  | 8   | 1.下次请交`ipynb`文件； 2.有一定注释； 3.可以考虑使用`os`库统筹安排文件    |
| 2021200641 | 林亦隆 | 2021经济学3班       | 6     |     |       |     |
| 2021200631 | 方至成 | 2021经济学-数学双学位实验班 | 6     |     |       |     |



- **Key Points for Score**

  - **Complete the base task:** Ensure your code can generate application letters for 30 universities * 3 majors.
  - **Follow the workflow:** We expect you to follow the workflow strictly.
  - **Original work:** DO NOT copy any code from your classmates.

- **Code Guide:** Refer to the Google Python Style Guide for elegant code writing

  - Chinese version: https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules.html
  - English version: https://github.com/google/styleguide
  - Sargent Code: https://github.com/Quant-of-Renmin-University/QuantEcon2025_RUC/blob/main/Codes/A_QuantEcon_Sargent/15_writing_good_code.ipynb

- **Tips for HW2**

  - **Use relative paths:** Instead of absolute paths, use relative paths to make your code more flexible and easier to maintain.

  - **Meaningful variable names:** Use proper and descriptive variable names that convey their purpose, which will definitely make it easier for team collaboration and others to understand your code. Try your best to avoid using generic names like `xxx1`, `xxx2`, etc.

  - **Writing comments:** Add comments to your code to make it more readable and understandable. Be careful that we always leave a space after the symbol `#` for a full line of comments. And Symbol `#` often need to be four spaces away from the code, for comments after the current line of code. Here is an example.

    - ```python
      # 这是一整行的注释
      print("HELLO WORLD!")
      
      print("DAY DAY UP!")    # 这是行内注释
      
      def calculate_add(x, y):
      """对两个数进行相加
      
      Args:
      	x (int): 第一个加数
      	y (int): 第二个加数
      
      Returns:
      	int: 两个数的和
      """
      	return a + b
      ```

  - **File process:** Use `os` to do file or path work. To get the complete path, we tend to connect via `os` to automatically match the differences in paths between systems, such as `Windows`, `Linux`, `Mac`. Before reading or writing a file or path, you can use `os.path.exists(path)` or `os.path.isfile(path)` to ensure it exists.

  - **Provide your output feedback:** Use `print` statements or `tqdm` to provide feedback on the progress of your loops, making it easier to understand how your code is executing. You can easily identify endless loops.

  - **Separate commands and code:** In `.ipynb` files, keep commands (e.g., `!pip install xxx` or `%pip install xxx`) and Python code in separate cells to maintain organization and clarity. You could also make `markdown` cells to make your code structure clearer and more readable.

