# Exploratory Data Analysis, Pt. 2
In the [last part](https://pabloagn.com/guided-projects/exploratory-data-analysis-pt-1/) of this 3-segment [Guided Project](https://pabloagn.com/guided-projects) we introduced the concept of **Exploratory Data Analysis** (*EDA*). We also stated a simple business case requested by our client, an insurance company, and proceeded to analyze a data set which was provided to us. We made some initial data exploration and chose a set of risk factors which could be potentially used to predict the severity of a given patient's Lung Cancer condition. 

In this section, we will go over 9 different classification algorithms. We will start by preparing our data. We will then implement each method step-by-step, and finally we'll make a performance comparison.

We'll be using Python scripts which can be found in the [Guided Project Repo](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis).

The generated plots and test results from last segment can also be found in the [plots](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis/plots) and [outputs](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis/outputs) folder respectively.

---

## Table of Contents
- Classification model design
	- Selecting our methods
	- Preparing the data
	- Logistic Regression
	- Decision Tree
	- Random Forest
	- Support Vector Machine
	- K-Nearest Neighbors
	- K-Means Clustering
	- Gaussian Naïve Bayes
	- Stochastic Gradient Descent
	- Kernel Approximation
	- Method comparison
- [Conclusions](#conclusions)
- [References](#references)

---

## Predictive model design
There are multiple models we can implement to try to predict the severity of Lung Cancer for a given patient. It's always a good idea to test at least a couple of different models and compare their accuracy.

Since we have categorical ordinal variables, there are three simple models we can use:












## Conclusions


## References
- https://www.kaggle.com/code/ekami66/detailed-exploratory-data-analysis-with-python/notebook
- https://www.lung.org/lung-health-diseases/lung-disease-lookup/lung-cancer/resource-library/lung-cancer-fact-sheet

