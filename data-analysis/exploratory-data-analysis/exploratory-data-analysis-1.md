# Exploratory Data Analysis, Pt. 1
A **Big Data file format** is designed to store high volumes of variable data optimally. This can be achieved using different formats, such as columnar or row-based.

Columnar formats store data by clustering entries by column, whereas row-based formats store data by clustering entries by row. Both formats are widely used in Big Data and present advantages & disadvantages among each other.

We can also further classify formats as text files or binary files. A binary file is designed to be read by computers; we cannot open a binary file and read its content simply using a text editor. In contrast, a text file can be directly opened with a text editor.

In this 3-article series, we will discuss six popular Big Data file formats, explain what they are for, go over writing & reading examples, and make some performance comparisons.

We'll be using Python scripts which can be found in the [Blog Article Repo](https://github.com/pabloagn/blog/tree/master/big-data/6-big-data-file-formats-compared).

---

## Table of Contents
- [Overview](#overview)
	- [Non-serialized formats](#1-non-serialized-formats)
		- [CSV](#11-csv)
		- [TXT](#12-txt)
	- [Serialized formats](#2-serialized-formats)
		- [Feather](#21-feather)
		- [Parquet](#22-parquet)
		- [Avro](#23-avro)
		- [Pickle](#24-pickle)
- [Creating a Data Set](#creating-a-data-set)
- [Writing with Python](#writing-with-python)
	- [CSV](#1-csv)
		- [Using numpy.tofile()](#11-using-numpytofile)
		- [Using  numpy.savetext()](#12-using-numpysavetext)
		- [Using pandas.DataFrame.to_csv()](#13-using-pandasdataframeto_csv)
	- [TXT](#2-txt)
		- [Using numpy.savetext()](#21-using-numpysavetext)
		- [Using pandas.DataFrame.to_csv()](#22-using-pandasdataframeto_csv)
	- [Feather](#3-feather)
		- [Using pandas.DataFrame.to_feather()](#31-using-pandasdataframeto_feather)
	- [Parquet](#4-parquet)
		- [Using pandas.DataFrame.to_parquet() without partitioning](#41-using-pandasdataframeto_parquet-without-partitioning)
		- [Using pandas.DataFrame.to_parquet() with a single partition](#42-using-pandasdataframeto_parquet-with-a-single-partition)
		- [Using pandas.DataFrame.to_parquet() with multiple partitions](#43-using-pandasdataframeto_parquet-with-multiple-partitions)
	- [Avro](#5-avro)
		- [Using fastavro Python file handler](#51-using-fastavro-python-file-handler)
	- [Pickle](#6-pickle)
		- [Using .pickle.dump() to write as an open file](#61-using-pickledump-to-write-as-an-open-file)
		- [Using .pickle.dumps() to write as a byte string](#62-using-pickledumps-to-write-as-a-byte-string)
- [Conclusions](#conclusions)
- [References](#references)

---

## Overview

### 1. Non-serialized formats
As opposed to the serialized formats, **non-serialized** formats do not convert the object into a stream of bytes. We will explain serialization formats in more detail further on. The most common non-serialization formats are CSV & TXT files.

#### 1.1 CSV
**Comma-separated values** (_CSV_) is a delimited text file format that typically uses a comma to separate values, and although more delimiters can be used, it's not standard practice. It is the most popular format for storing & reading tabular data since it's fast, easy to write & read, supported by practically all programs & libraries involving data processing, and forces a flat & simple schema.

As popular as it is, CSV also has some disadvantages, such as large file sizes, slow parsing time, poor support from Apache Spark, missing data handling, limited encoding formats, special handling required with nested data, basic data support only, lack of support of special characters, no defined schema, and the use of commas as delimiters; if our data entries have commas, we will have to enclose the entry in quotes. Otherwise, they will be treated as delimiters.

These disadvantages make CSV files suboptimal when working with big data.

A typical CSV file will have a `.csv` extension and will look like the example below:

```
Name,Age,Occupation,Country,State,City
Joe,20,Student,United States,Kansas,Kansas City
Chloe,37,Detective,United States,California,Los Angeles
Dan,39,Detective,United States,California,Los Angeles
...
```

Some considerations:
- The header is denoted as the first row of our document.
- Each entry is followed by a comma but without blank spaces.
- Entries can have blank spaces and will be treated accordingly when parsing.
- Even though we can use text and numeric values, a CSV file will not store information regarding data types.

#### 1.2 TXT
**Text document file** (_TXT_) is a plain-text format structured as a sequence of lines of text. It is also a prevalent format used for storing & reading tabular data because of its simplicity & versatility; a TXT file can be formatted as delimited, free form, fixed width, jagged right, and so on.

A typical TXT file will have a `.txt` extension and will look like the example below _(depending on the delimiter used, it will vary. In this example, we use tab delimiters which is the convention)_: