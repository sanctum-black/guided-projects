# Exploratory Data Analysis, Pt. 3

[![made-with badge](https://img.shields.io/static/v1?label=Made%20with&message=Obsidian&color=7d5bed&logo=obsidian&labelColor=1a1a1a&style=flat)](https://obsidian.md/)

[![type](https://img.shields.io/static/v1?label=Type&message=guided-project&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAAi0lEQVRIS+2WMQ7AIAhF/UNXrtP7rz2OYxeqTWxMTBUSxQVXfnzyQQKC8YExL7zAGCNbgIkIsIKVhBw4vbR7unR6Gp0LvwxXd2v+EvkdDpxWXpWlRTyi9/pABRyBJHEHSlxSadxSlV0SsVsqcUml2W/pynWxnsXNisHMRxrCl8qvH3ECnQDuOmy+0zwB4WNxmUKgwwAAAABJRU5ErkJggg==&labelColor=1a1a1a&style=flat)](https://pabloagn.com/guided-projects/) [![category](https://img.shields.io/static/v1?label=Category&message=data-science&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAB9UlEQVRIS6VWMU7DQBDkDAQEdrAoCISCAomCL1DxC95Azy9oeQS/oOIHVFAgREFoCHGCRSzZzEU+63LZ9W6CO/vudmZ2d9Zn1pTPaDSqut2usduHw+FpFEUv7t1fk8LNAkiPDWj3+ADuTPjNvXMxWwGzLCuqqtqwh5MkiY0xEwfOAfrEKFAWUBO4DZQDXgCEqjuouvbZUanUrocpngMMVUkKtKC+WhFQUudAUd8r1PkepJ/w7Tysn4uzkNJlascF9WOASAki6w0xrn19b3Gpps5y3kRfJADPZgr9gJSP0EgDHDiQ/Mp50PfxAmDtuQhsZmb/z0OVhwSkmGrSGp5bGRDp3EFaJ5JaiahdZ2vYNj/JkWVMgW7sgNw2yOW+99gacp7TeFE72OcUrgo4Ho93+/3+D5T9QmGHm0BNSnHgMI7jj9Ai2tElZGCK9S3S+GA4BcNNydBaIuEstu/iLJWCa+pLDm+Nz+xQAsBenucnRVG8asFq0s/Yf9YoVAI21wyn3N4I7M1A8ijWHwB42XrFqIO9YfMRlVqqyXC5ukED3nIEVJcoBXv1lmWa5gIpeeQioyTWVj1uXf0DpgKUZbmfpunXKnVnU9rWDKiTHRSDNkDu36iqIQK/Q+mxU8sBYniL/1EVoJ9Wqwo/5x6Cf9YKv6Em1XbNH5bGfSwvuRe1AAAAAElFTkSuQmCC&labelColor=1a1a1a&style=flat)](https://pabloagn.com/categories/data-science/) [![technologies](https://img.shields.io/static/v1?label=Technologies&message=LaTeX,Texmaker&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAACXBIWXMAAAsTAAALEwEAmpwYAAAA1klEQVR4nM2RMW7CUBBEnUikIQUIlBJxrrQgJG7ABRBnoUkaWhpoUgWJlgNYbvz/G1dUi1ayoy87rpOtVrszs6OdLPtXlef5UNJXjHHcCwohjMzsKZ3FGN+Bq/e+c0xHGfiWtEznkg6SNnW/dIxjs0YJ2AMnM3tJSFPgHkKY17gBcAQ+zOw5A3aSbsCkdW0NnNOZY2rstpcInJ3cS/SzwGdqtSzLmdusquqtIXWsehVF8QpcJK1qmxt/TMv6wjE/z0leP27i8Ag8inT/axxtAQ+9o/zn9QD3JOiyTjnQEQAAAABJRU5ErkJggg==&labelColor=1a1a1a&style=flat)](https://pabloagn.com/technologies/) [![website article](https://img.shields.io/static/v1?label=Website&message=Post%20Link&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAACXBIWXMAAAsTAAALEwEAmpwYAAAB+ElEQVR4nO2VOYgUURCGR/BAI4MN1EwjI89EMDYQvNBNNNlcE0VBUdlUUSMjj2BF2UDRePDAwGzNF2GNPIYd8Hjv/6YnEHSf/FIDPTJiu4nJFBTd1Kv6/nrVBd1q/S8DJiU9AmaBm5LOSjoATPwDY0LSQUnnzDArmJOjkqclvQceSHohaR6oJC1JeiPprqT9pZSVg5pSyirH4sw5S1EzbwZwP5jTIwWBdj1meEppZ6/XOyXpCdCX9Am4Fv45Yo+Bk1VV7ag3FNz2kKC7yznvHiX4u3U6nXU55xPAW7vfHfvLmNtmW8NaFux67k0Ea03esTfJJQTj23bHgiNtPNK6jZem3Wpg46Wp23hp2q0GNl6axksjaRGYkXRF0mnHq6ra2HSk/X5/k6RDks6YEazFPwnuBS5KuirptqTnkj4CJZ4zwNFSytqBoP/2wDHgXi33A/BM0i2zzDR7SBC4LGlPr9fb5huVUlYMus45b5E0FYJfgQS8C8/Al7jJVEpp86DODLPMNDs0up7xXBQZVKLLb8CCpIfA+ZzzvpTS+lLKGuAI8DT8cClltc+c49yoWQjGL140ao25oW8QXW1IKe3KOR8Hbkh66ZtI+i7plaG+iR244JjP3HDkXnetGWbVp9XYopHtHgvwWtIPu9+BSx7bssBNDdhqX07xT/Jbz1SBBDGHAAAAAElFTkSuQmCC&labelColor=1a1a1a&style=flat)](https://pabloagn.com/guided-projects/exploratory-data-analysis-pt-3/)

Over the [last two parts](https://pabloagn.com/guided-projects/exploratory-data-analysis-pt-2/) of this 3-segment [Guided Project](https://pabloagn.com/guided-projects), we have introduced some interesting concepts around EDA. We've performed statistical analyses on our client's data set and evaluated multiple classification algorithms. We started with a hypothesis, put it to test, refined our beliefs, came up with some conclusions, and now, it's time to face our client.

In this section, we will learn how to build a business client deliverable using [LaTeX](https://pabloagn.com/technologies/latex/) and [Texmaker](https://pabloagn.com/technologies/texmaker/). We will translate all our results into an elegant report which will help with decision execution. We will close this Guided Project with some recommendations on discussing technical results with a non-technical audience.

The generated plots and test results from the last two segments can be found in the [plots](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis/plots) and [outputs](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis/outputs) folder respectively.

---

## Table of Contents
- [Consolidating the results](#consolidating-the-results)
- [Generating a technical reference deliverable](#generating-a-technical-deliverable)
	- [Managing expectations](#1-managing-expectations)
	- [An introduction to LaTeX & Texmaker](#2-an-introduction-to-latex--texmaker)
	- [Preparing our environment and downloading a template]()
	- [Designing our layout](#3-designing-our-layout)
		- [Executive summary](#31-executive-summary)
		- [Business guide](#32-business-guide)
		- [Plot results](#33-plot-results)
		- [Tabular results](#34-tabular-results)
		- [Method considerations & limitations](#35-method-considerations--limitations)
		- [Conclusions & recommendations](#36-conclusions--recommendations)
		- [Appendix](#37-appendix)
- [Conclusions](#conclusions)
- [References](#references)

---

## Consolidating the results


## Generating a technical deliverable

### 1. Managing expectations
Before jumping right in we need to have something clear: We might have a high-performing model presenting extremely high accuracy results. Still, if we don't communicate our results properly and translate them into actionable insights, there's a chance that the client won't understand the value behind our hard work.

Although things have changed drastically during the last years, it's common for companies to still not have an established team of data scientists and data analysts, specially when talking about small emerging businesses; managing an internal IT department is already expensive, so there is no assurance that our client will have a dedicated research team behind the curtains. And even if this is the case, there is no guarantee that they will be available for our project.

This is why we need to ensure that we're translating our results to a business-oriented language and not a research-oriented one as a complement to our technical deliverable, specially if we are presenting our findings to a commercial team, which is often the case.

Thankfully, there are some really nice tools out there that will help us achieve our goal.

### 2. An introduction to LaTeX & Texmaker
LaTeX is a software system for document preparation based on TeX, which can render complex mathematical formulae and multiple objects such as tables, headers, images, plots and other figures. It's the go-to language for scientific and technical journaling because of its flexibility, active community and beautiful document generating capabilities.

There are multiple TeX distributions for different applications, the most common ones being TeX Live, MiKTeX, and MacTeX. They differ mainly in their package content and platform support.

Texmaker is a widely adopted, cross-platform, open-source LaTeX editor with an integrated build tool and PDF viewer.

LaTeX syntax is not easy; it presents a steep learning curve specially if we are to explore all of its functionalities and packages. The good news is there is a vast collection of templates for multiple purposes already available. Such templates usually include a LaTeX source file `.tex`, a LaTeX class definition file `.cls`, a compiled `.pdf` example, and required digital assets such as `.png`, `.jpg` or `.pdf` files.

The nice thing about templates, is that they're fully customizable; we can download a `.cls` class file and tailor it to our needs until we have exactly what we're looking for. At first it takes some time, but in the end it's really worth it.

### 3. Preparing our environment and downloading a template
For this segment we will use the TeX Live distribution along with the Texmaker editor. We will also download a template from [LaTeX Templates](https://www.latextemplates.com/template/sullivan-business-report), a phenomenal website providing free, fully-fledged material.

This tutorial will be specifically oriented towards Windows, but can easily be tailored for macOS or Linux operating systems.

We will start by installing our **Tex Live** distribution:
- Head to the [Tex Live official website](https://www.tug.org/texlive/).
- Select *install on Windows* from the *Concise instructions, per platform* section.
- Head to the [Easy install](https://www.tug.org/texlive/windows.html#install) section.
- Download the `install-tl-windows.exe` executable and run it on your machine. (*Please make sure to read the official documentation carefully before installing*).
- Wait for the installation process to complete (*depending on the mirror selected, it could take several minutes, so make sure a mirror close to your current location is selected*).

We will then install **Texmaker** for Windows:
- Head to the [download page](https://www.xm1math.net/texmaker/download.html).
- Select the *Desktop msi installer for windows 7/8/10/11 64 bits* package.
- Run the executable on your machine.

Upon conclusion, we should end up with the Tex Live distribution and the Texmaker application installed.

We will then download our template:
- Head to the [Arsclassica Article](https://www.latextemplates.com/template/arsclassica-article) template page.
- We can view the PDF preview by heading to the *Preview Template PDF* section. This is what our actual document will look like out of the box, but not to worry, we will fully customize it.
- Click the *Download Template Code* link.
- A `.zip` file will download. We will then need to extract its contents.
- Once we're done, we can create a new directory named `client_report` and paste all of our template's contents.
- We should have the following:
	- `Figures`
	- `article_4.tex`
	- `sample.bib`
	- `structure.tex`

We will open our template in Texmaker to make sure everything's working fine:
- Open the Texmaker application.
- Head to *File*, *Open* and select the `article_4.tex` file.
- We should end up with the `.tex` file displaying on the left panel, and a blank canvas on the right panel. This canvas is where the `.pdf` file will be displayed once we compile our document.
- To compile our document, head to *Quick Build* option on the top panel and select *run*.
- Our document will be compiled for the first time, and if everything went fine, two things will happen:
	- A `.pdf` file along with several other files will be generated in our working directory.
	- The compiled file will appear on the right panel.
- LaTeX documents have to be re-compiled each time we want to visualize our final `.pdf` document. The good thing is that this is going to take less time on next iterations.

We could use this template as-is by simply editing our `article_4.tex` file. This would not be a bad idea since our template has almost everything we require out-of-the-box. Still, we want to modify certain aspects of our document to make it truly our own. For that, we will need to locate our `structure.tex` file and open it in any text editor. [VS Code](https://pabloagn.com/technologies/vs-code/) offers a nice extension we can [download here](https://marketplace.visualstudio.com/items?itemName=torn4dom4n.latex-support) for LaTeX language support. This will enable syntax highlighting, formatting and other useful features that will make it easier for us to manipulate `.tex` files.

We are now ready to start customizing our template.

### 3. Designing our layout
If we take a close look at our `structure.tex` file, we can see that it's already nicely divided by sections.

The first section imports the required packages:

##### **Code**
```LaTeX
\usepackage[
nochapters, % Turn off chapters since this is an article        
beramono, % Use the Bera Mono font for monospaced text (\texttt)
eulermath,% Use the Euler font for mathematics
pdfspacing, % Makes use of pdftex’ letter spacing capabilities via the microtype package
dottedtoc % Dotted lines leading to the page numbers in the table of contents
]{classicthesis} % The layout is based on the Classic Thesis style

\usepackage{arsclassica} % Modifies the Classic Thesis package

\usepackage[T1]{fontenc} % Use 8-bit encoding that has 256 glyphs

\usepackage[utf8]{inputenc} % Required for including letters with accents

\usepackage{graphicx} % Required for including images
\graphicspath{{Figures/}} % Set the default folder for images

\usepackage{enumitem} % Required for manipulating the whitespace between and within lists

\usepackage{lipsum} % Used for inserting dummy 'Lorem ipsum' text into the template

\usepackage{subfig} % Required for creating figures with multiple parts (subfigures)

\usepackage{amsmath,amssymb,amsthm} % For including math equations, theorems, symbols, etc

\usepackage{varioref} % More descriptive referencing
```

The next section defines theorem styles:

##### **Code**
```LaTeX
\theoremstyle{definition} % Define theorem styles here based on the definition style (used for definitions and examples)
\newtheorem{definition}{Definition}

\theoremstyle{plain} % Define theorem styles here based on the plain style (used for theorems, lemmas, propositions)
\newtheorem{theorem}{Theorem}

\theoremstyle{remark} % Define theorem styles here based on the remark style (used for remarks and notes)
```

The last section defines styling properties for hyperlinks:

##### **Code**
```LaTeX
\hypersetup{
%draft, % Uncomment to remove all links (useful for printing in black and white)
colorlinks=true, breaklinks=true, bookmarks=true,bookmarksnumbered,
urlcolor=webbrown, linkcolor=RoyalBlue, citecolor=webgreen, % Link colors
pdftitle={}, % PDF title
pdfauthor={\textcopyright}, % PDF Author
pdfsubject={}, % PDF Subject
pdfkeywords={}, % PDF Keywords
pdfcreator={pdfLaTeX}, % PDF Creator
pdfproducer={LaTeX with hyperref and ClassicThesis} % PDF producer
}
```

This is actually a really simple & straightforward template, mostly due to the fact that it's using the `classicthesis` as underlying layout.

We will start by making some modifications to our structure:

##### **Code**
```LaTeX
Aa
```




#### 3.1 Executive summary


#### 3.2 Business guide


#### 3.3 Plot results


#### 3.4 Tabular results


#### 3.5 Method considerations & limitations


#### 3.6 Conclusions & recommendations


#### 3.7 Appendix





## Conclusions


## References
- A
- A

