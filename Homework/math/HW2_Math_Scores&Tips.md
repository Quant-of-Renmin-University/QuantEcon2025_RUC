| StudentID  | Name     | number | group | HW2: Score | Some Tips                                                    |
| ---------- | -------- | ------ | ----- | ---------- | ------------------------------------------------------------ |
| 2022202681 | 曹越晰   | 0      | 1     | 9.5        | 1.  一般读取excel文件进行【数据处理】，用`pandas`比较多，`openpyxl`一般更多用于精细的excel操作（比如设置excel单元格格式等）；`pandas`则在数据处理工作中占优； |
| 2022202710 | 邓双贤   | 1      | 1     | 9          |                                                              |
| 2022201343 | 刘倡源   | 2      | 1     | 8          | 1.  AI生成气比较重，读取excel转为`DataFrame`一般使用的方式是`pd.read_excel`即可；可以课后查一下`pd.read_excel`【函数】与`pd.ExcelFile`【类】的区别 |
| 2022202743 | 马瑜梓   | 3      | 1     | 7.5        | 1. 建议同学生成完看一下pdf文档是否格式有问题；2. 我们希望同学能够严格按照workflow来进行操作 |
| 2022201412 | 徐子禾   | 4      | 2     | 7.5        | 1. 下次作业上传`ipynb`文件；2. 一些没必要定义的函数可以简化  |
| 2021200657 | 梁艺高   | 5      | 2     | 9          |                                                              |
| 2022202747 | 郭立为   | 6      | 2     | 8          | 1. 文件路径的合并一般使用`os`库；2. `for`循环所有的信息的时候我们一般会提前定义行数和列数（用变量表示[函数`df.shape` ->  一个`tuple`元组]）以免数量不固定还需要修改代码 |
| 2021202449 | 曹馨元   | 7      | 2     | 9          |                                                              |
| 2022201353 | 俞项天   | 8      | 3     | 8.5        | 1. 可以加一些注释；另外=, <这些符号左右两边都会有一个空格，注意这些点可以提高代码可读性 |
| 2022202654 | 江孟书   | 9      | 3     | 8.5        | 1. 下次上传`ipynb`文件                                       |
| 2022201372 | 胡熙媛   | 10     | 3     | 8          | 1. 下次上传`ipynb`文件；2. 按要求上传作业； 3. 不必要的ai生成的注释可以删去 |
| 2022201480 | 梁泓铭   | 11     | 3     | 8.5        | 1. 可以输出一些中间结果来测试程序正常运行                    |
| 2022201462 | 欧阳语博 | 12     | 4     | 7.5        | 1. 格式：注释# 一般后面会有一个空格；2. 作业要求使用`docxtpl`库；3. 下次上传`ipynb`文件 |
| 2022202610 | 程忆楠   | 13     | 4     | 8          | 1. 在`ipynb`cell中如果需要运行`cmd`/终端命令，windows系统需要前面加上`!`，linux系统需要加上`%`； |
| 2022202625 | 陈尚祺   | 14     | 4     | 8          | 1. `ipynb`文件中可以选择每个单元格的格式，我们要选择python才可以运行 |
| 2022202763 | 谢丽媛   | 15     | 4     | 8.5        | 1. 所有的`import`命令一般在code最开始的时候呈现              |
| 2022202672 | 邵远平   | 16     | 5     | 9.5        |                                                              |
| 2022202655 | 方国圳   | 17     | 5     | 8.5        | 1. 注意注释的格式，一般`# `后面需要有一个空格                |
| 2021200691 | 滕明阳   | 18     | 5     | 9          | 1. 注意注释的格式，一般`# `后面需要有一个空格；p.s. SOP中[your university] |
| 2022202709 | 李馨怡   | 19     | 5     | 9.5        |                                                              |
| 2021201687 | 朱堃琳   | 20     | 6     | 8          | 1. 可以善用jupter notebook的cell；2. `import`命令一般在文件开头声明 |
| 2022202620 | 王成林   | 21     | 6     | 7.5        | 1. 按照workflow来做作业                                      |
| 2022202692 | 唐汇宸   | 22     | 6     | 8.5        | 1. 下次请上传`ipynb`文件；2. 注意一下上面同学也提到的`pip install`的命令 |
| 2022202619 | 张沛渊   | 23     | 6     | 8.5        |                                                              |



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

