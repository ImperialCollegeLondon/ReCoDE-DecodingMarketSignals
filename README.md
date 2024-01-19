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
| Reading    | 3 hours |
| Practising | 5 hours |

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
- Familiarity with Jupyter notebooks

### Academic

<!-- List the system requirements and how to obtain them, that can be as simple
as adding a hyperlink to as detailed as writting step-by-step instructions.
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

The repository is self-contained. Additional references are provided in teh Jupyter notebooks. 


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

A recent mid-class laptop is sufficient. The code was developed on a Linux machine. The `shell` script written
can be translated to work with MacOS and can be modified to work on Linux using the principles explained.

We make use of `Python 3.11`. If you are comfortable with an older version of Python, library we need to make run 
`ta-lib` was tested to be installable from `pypi` on `Python v.3.8` and `3.9`.
If you just want to get started, use `Python v. 3.8`. `ta-lib` can then be installed using `pip install TA-Lib`. 
If you want to make use of later versions of Python such as the environment this project was developed on, precisely 
`Python v. 3.11`, the process is more involved and requires compiling `C++` files from source. 


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
Once you have an understanding of the overall goal, you can start either replicating the results, 
or applying them to your own data. For that, you need to set up your Python environment. The reader is encouraged to apply the methods 
to their own data from different markets, for instance, the futures or forex market. 


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
