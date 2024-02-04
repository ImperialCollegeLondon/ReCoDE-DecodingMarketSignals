<!-- Your Project title, make it sound catchy! -->

# Decoding Market Signals: Leveraging candlestick patterns, machine learning and alpha signals for enhanced trading strategy analysis

<!-- Provide a short description to your project -->

## Description

This project aims to rigorously back-test a trading strategy, focusing on evaluating the informational value of technical trading signals, particularly candlestick patterns. Utilizing Python, an automated pipeline will systematically scan the market, collecting publicly available data for analysis. Advanced functionalities of the Pandas library will be employed for detailed statistical characterizations and efficient data storage.

The project's core involves assessing the predictive capabilities of these trading signals using nuanced binary classification performance metrics, thereby determining their practical applicability. Additionally, a logistic regression model will be deployed to explore the intersection of finance and machine learning. This phase aims to ascertain whether machine learning algorithms can outperform traditional methods in predicting market movements based on identified signals.

This multifaceted project integrates financial analysis, data science, and machine learning, promising valuable insights with both academic and practical implications. Its methodologically sound approach, coupled with detailed documentation and learning annotations, is designed to make it an exemplary contribution to the ReCoDE initiative, showcasing the transformative potential of research computing and data science in diverse disciplines.

<!-- What should the students going through your exemplar learn -->

## Learning Outcomes

- Setting up a custom computational environment for financial data science.
- Making a custom technical analysis library written in C++ work with recent Python. 
- Obtaining and pre-processing high-quality.
- Using Pandas' best-practices like method-chaining, and multi-index data frames for data manipulation  
- Independently testing and analysing trading actions proposed on a hypothesis.
- Learning an approach "from paper to code": We will translate parts of a densely-written paper that claims to contain "alpha signals" to Python code and analyse whether they work in practice.


<!-- How long should they spend reading and practising using your Code.
Provide your best estimate -->

| Task       | Time    |
| ---------- | ------- |
| Reading    | 4 hours |
| Practising | 6 hours |

## Requirements

<!--
If your exemplar requires students to have a background knowledge of something
especially this is the place to mention that.

List any resources you would recommend to get the students started.

If there is an existing exemplar in the ReCoDE repositories link to that.
-->

- Foundational knowledge of Python
- An interest in stock markets and trading signals
- An interest in statistical analysis and hypothesis testing
- Resilience in troubleshooting and adapting older libraries to work with recent Python versions.
  - Particularly, we will make use of a library called `ta-lib` that contains a pattern-recognition library detecting candlestick patterns in Open-High-Low-Close `(OHCL)` data.
- Familiarity with Jupyter notebooks.

### Academic

<!-- List the system requirements and how to obtain them, that can be as simple
as adding a hyperlink to as detailed as writing step-by-step instructions.
How detailed the instructions should be will vary on a case-by-case basis.

Here are some examples:

- 50 GB of disk space to hold Dataset X
- Anaconda
- Python 3.11 or newer
- Access to the HPC
- PETSc v3.16
- gfortran compiler
- Paraview
-->

The repository is self-contained. Additional references are provided in the Jupyter notebooks. 


### System

<!-- Instructions on how the student should start going through the exemplar.

Structure this section as you see fit but try to be clear, concise and accurate
when writing your instructions.

For example:
Start by watching the introduction video,
then study Jupyter notebooks 1-3 in the `intro` folder
and attempt to complete exercise 1a and 1b.

Once done, start going through through the PDF in the `main` folder.
By the end of it you should be able to solve exercises 2 to 4.

A final exercise can be found in the `final` folder.

Solutions to the above can be found in `solutions`.
-->

A recent mid-class laptop is sufficient. The code was developed on a Linux machine. 

In this code exemplary, we make use of `Python 3.11`. 
As already hinted, we identify the candlestick patterns in financial markets data using a library that is called `ta-lib`.
It works well for our task, but is no longer maintained. If you are comfortable with an older version of Python, precisely 
`Python 3.8` or `Python 3.9`, or just want to get started, it is straightforward to install the library using `pip` or `conda`. 

`ta-lib` was tested to be installable from `pypi` on `Python 3.8` and `3.9`.
If you just want to get started, use `Python 3.8`. `ta-lib` can then be installed using `pip install TA-Lib`. 
Alternatively, if you want to make use of the `conda` package manager, use `conda install ta-lib`
For `Python 3.9`, the author observed on a Linux operating system, that `conda install ta-lib` worked straightforward, whereas `pip install TA-Lib` did not.

If you want to make use of later versions of Python such as the environment this project was developed on, precisely 
`Python v. 3.11`, the process is more involved and requires compiling `ta-lib's C++` files from source.

Ww know met two common problem of computer scientists, data scientists and practitioners:
i) Making `legacy` code run on modern systems,
ii) Facing multiple choices of what package manager to use.
If you are just interested on getting started, use `Python 3.8` and skip the following section.

For the interested reader, is follows some background information on i) and ii).
Issue i) is commonly encountered in practice, especially in larger corporations or custom software whose author's stopped 
maintaining their code. There is no silver bullet on working with legacy code and custom problem often need tailor-made solutions.
For this project, the author wrote
a `shell` script that is custom-made to set up `ta-lib` for `Python 3.11`. Whenever you are not sure whether a solution works and you
have reasons to believe your attempt is error-prone, might have side-effects, or spoil the operating system, it is advisable 
to work from within a virtual environment, for example using `Docker`, before employing a working solution on the user's machine. 
Discussing `Docker` is beyond the scope of this documentation.

The `shell script` can also be directly applied to work on MacOS as the latter uses the `Z shell` by default. 
The `Z shell`, is also known as `zsh` and is a `Unix shell` that is built on top of `bash`. Hence, compatibility 
is likely and the script should run without reservation. 
For Windows users the `shell script` can also be modified to work on the `Windows shell` or `PowerShell`.
The equivalent of a Linux `shell script` on Windows is a `batch script` and the commands expressed have to be translated 
to make them compatible on Windows.

Let us now quickly address issue ii):
If your Python environment is set up using miniconda (recommended), see also `https://docs.conda.io/projects/miniconda/en/latest/`,
both, `conda` and `pip` are installed by default, and you can make use of both them.   
If you Python environment is set up using the source files from `https://www.python.org/` you might have to install `pip` separately
and cannot use the benefits of conda.

What then is the difference between `pip` and `conda`?
`Pip` is a package manager specifically designed for Python packages.
It primarily focuses on installing and managing Python libraries and packages from the `Python Package Index (PyPI)`.
Pip is used for managing Python dependencies within a Python environment.
On the other hand, `Conda` is a more comprehensive package manager and environment manager.
While it can manage Python packages, it is not limited to Python and can handle packages and libraries from various programming languages.
Conda is often used to create isolated environments that can include different versions of Python and non-Python dependencies.
It can manage both Python packages and system-level packages and is capable of handling complex dependency resolution.

Managing and detecting version conflicts of a large Python setup again, is a topic on its own. Granted `conda` is diligent, but slow, the 
reader is encouraged to look into promising alternatives like `mama`, which is a package manager written in `C++` and hence more performant 
than conda, although less tested.

A final note regarding code-formatting. To comply with the PEP-8 style guide for Python code,  
`https://peps.python.org/pep-0008/`, we make use of a code-formatter, that automatically spots issues concerning spacing
and style. It is applied on code that runs error-free and ensures style consistency. There are several open-source code-formatters
out there and arguably the most popular are `black`, see `https://github.com/psf/black`, and `Ruff`, see `https://docs.astral.sh/ruff/formatter/`.
The former is well-tested, however the latter is more performant and has recently gained increasing attention.


## Getting Started

<!-- An overview of the files and folder in the exemplar.
Not all files and directories need to be listed, just the important
sections of your project, like the learning material, the code, the tests, etc.

A good starting point is using the command `tree` in a terminal(Unix),
copying its output and then removing the unimportant parts.

You can use ellipsis (...) to suggest that there are more files or folders
in a tree node.
-->

Start by opening and reading through the Jupyter notebook. All essential steps are separated in respective sub-sections.
Once you have an understanding of the overall goal, you can start setting up your Python environment along `ta-lib and 
either replicate the results or apply the techniques demonstrated to your own data. The reader is encouraged to apply the methods outlined 
on their own data from different markets, for instance, the futures and forex markets. 

<!--
The below is a TODO
-->

[comment]: <> (## Project Structure)


[comment]: <> (```log)

[comment]: <> (.)

[comment]: <> (├── examples)

[comment]: <> (│   ├── ex1)

[comment]: <> (│   └── ex2)

[comment]: <> (├── src)

[comment]: <> (|   ├── file1.py)

[comment]: <> (|   ├── file2.cpp)

[comment]: <> (|   ├── ...)

[comment]: <> (│   └── data)

[comment]: <> (├── app)

[comment]: <> (├── docs)

[comment]: <> (├── main)

[comment]: <> (└── test)

[comment]: <> (```)

<!-- Change this to your License. Make sure you have added the file on GitHub -->

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
