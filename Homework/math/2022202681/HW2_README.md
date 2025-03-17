
# University Application Letter Generator

This project automates the generation of personalized application letters (Statement of Purpose) for graduate school applications. By leveraging Python, it efficiently generates Word and PDF files tailored to each university and program combination.

*Author: Yuexi Cao*|
*last update: Mar 06, 2025*

---

## Table of Contents
1. [Purpose](#purpose)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [Example Output](#example-output)
7. [Contributing](#contributing)

---

## Purpose
When applying to multiple graduate schools, writing individualized application letters can be time-consuming. This tool simplifies the process by:
- Using a single Word template (`application_template.docx`) with placeholders for `university_mame` and `program_name`.
- Reading a list of universities and programs from an Excel file (`application_list.xlsx`).
- Generating customized Word and PDF files for each university-program combination.

By automating this task, you save significant time and effort while ensuring consistency and accuracy in your application materials.

---

## Project Structure
The project is organized as follows:

```
HW2/
├── code/
│   └── generate_applications.py          # Python script to generate application letters
├── data/
│   ├── application_template.docx         # Template file with placeholders
│   └── application_list.xlsx             # Excel file containing university names and programs
├── result/
│   ├── word_files/                       # Generated Word files will be saved here
│   └── pdf_files/                        # Generated PDF files will be saved here
└── README.md                             # This file
```

---

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yuexi-cao/university-application-generator.git
   cd university-application-generator
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.x installed, then install the required libraries:
   ```bash
   pip install openpyxl docxtpl docx2pdf
   ```

3. **Prepare Input Files**:
   - Place your Word template in `data/application_template.docx`.
   - Update the Excel file `data/application_list.xlsx` with your target universities and programs.

4. **Run the Script**:
   Execute the Python script to generate the application letters:
   ```bash
   python code/generate_applications.py
   ```

---

## Usage
1. **Edit the Template**:
   - Open `data/application_template.docx` and customize the content to match your personal story.
   - Ensure the placeholders `university_name` and `program_name` are present.

2. **Update the Excel File**:
   - In `data/application_list.xlsx`, list the universities in the first column and their corresponding programs in the next three columns.
   - Leave empty cells if a university has fewer than three programs.

3. **Generate Files**:
   - Run the script to generate Word files in `result/word_files` and PDF files in `result/pdf_files`.

---

## Dependencies
The following Python libraries are required:
- `openpyxl`: For reading the Excel file.
- `docxtpl`: For filling the Word template.
- `docx2pdf`: For converting Word files to PDF (Windows only).

Install them using:
```bash
pip install openpyxl docxtpl docx2pdf
```

---

## Example Output
### Input
#### Excel File (`application_list.xlsx`):
| University Name | Program 1            | Program 2            | Program 3              |
|-----------------|----------------------|----------------------|------------------------|
| Harvard         | MA in Economics      | MA in Statistics     | PhD in Data Science    |
| Stanford        | MA in Economics      |                      |                        |
| MIT             | MA in Statistics     | PhD in Data Science  |                        |

#### Template File (`application_template.docx`):
```
Dear Admissions Committee,

I am writing to express my strong interest in pursuing {{program_name}} at {{university_name}}. As an undergraduate student majoring in Economics and Mathematics at Renmin University of China, I have cultivated a solid foundation in quantitative analysis, economic theory, and data-driven research.
...
Sincerely,
Yuexi Cao
```

### Output
#### Generated Word Files:
- `Harvard_MA_in_Economics.docx`
- `Harvard_MA_in_Statistics.docx`
- `Harvard_PhD_in_Data_Science.docx`
- `Stanford_MA_in_Economics.docx`
- ...

#### Generated PDF Files:
- `Harvard_MA_in_Economics.pdf`
- `Harvard_MA_in_Statistics.pdf`
- `Harvard_PhD_in_Data_Science.pdf`
- `Stanford_MA_in_Economics.pdf`
- ...

---

## Contributing
If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request. Contributions are welcome!

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Final Notes
This tool is designed to streamline the application process, saving you time and effort. Customize the template and input files to suit your needs, and let the script handle the rest!

Happy applying! 🎓✨

---
