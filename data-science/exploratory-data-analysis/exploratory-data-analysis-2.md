# Exploratory Data Analysis, Pt. 2

[![made-with badge](https://img.shields.io/static/v1?label=Made%20with&message=Obsidian&color=7d5bed&logo=obsidian&labelColor=1a1a1a&style=flat)](https://obsidian.md/)

[![type](https://img.shields.io/static/v1?label=Type&message=guided-project&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAAi0lEQVRIS+2WMQ7AIAhF/UNXrtP7rz2OYxeqTWxMTBUSxQVXfnzyQQKC8YExL7zAGCNbgIkIsIKVhBw4vbR7unR6Gp0LvwxXd2v+EvkdDpxWXpWlRTyi9/pABRyBJHEHSlxSadxSlV0SsVsqcUml2W/pynWxnsXNisHMRxrCl8qvH3ECnQDuOmy+0zwB4WNxmUKgwwAAAABJRU5ErkJggg==&labelColor=1a1a1a&style=flat)](https://pabloagn.com/guided-projects/) [![category](https://img.shields.io/static/v1?label=Category&message=data-science&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAB9UlEQVRIS6VWMU7DQBDkDAQEdrAoCISCAomCL1DxC95Azy9oeQS/oOIHVFAgREFoCHGCRSzZzEU+63LZ9W6CO/vudmZ2d9Zn1pTPaDSqut2usduHw+FpFEUv7t1fk8LNAkiPDWj3+ADuTPjNvXMxWwGzLCuqqtqwh5MkiY0xEwfOAfrEKFAWUBO4DZQDXgCEqjuouvbZUanUrocpngMMVUkKtKC+WhFQUudAUd8r1PkepJ/w7Tysn4uzkNJlascF9WOASAki6w0xrn19b3Gpps5y3kRfJADPZgr9gJSP0EgDHDiQ/Mp50PfxAmDtuQhsZmb/z0OVhwSkmGrSGp5bGRDp3EFaJ5JaiahdZ2vYNj/JkWVMgW7sgNw2yOW+99gacp7TeFE72OcUrgo4Ho93+/3+D5T9QmGHm0BNSnHgMI7jj9Ai2tElZGCK9S3S+GA4BcNNydBaIuEstu/iLJWCa+pLDm+Nz+xQAsBenucnRVG8asFq0s/Yf9YoVAI21wyn3N4I7M1A8ijWHwB42XrFqIO9YfMRlVqqyXC5ukED3nIEVJcoBXv1lmWa5gIpeeQioyTWVj1uXf0DpgKUZbmfpunXKnVnU9rWDKiTHRSDNkDu36iqIQK/Q+mxU8sBYniL/1EVoJ9Wqwo/5x6Cf9YKv6Em1XbNH5bGfSwvuRe1AAAAAElFTkSuQmCC&labelColor=1a1a1a&style=flat)](https://pabloagn.com/categories/data-science/) [![technologies](https://img.shields.io/static/v1?label=Technologies&message=Python&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAACXBIWXMAAAsTAAALEwEAmpwYAAAA1klEQVR4nM2RMW7CUBBEnUikIQUIlBJxrrQgJG7ABRBnoUkaWhpoUgWJlgNYbvz/G1dUi1ayoy87rpOtVrszs6OdLPtXlef5UNJXjHHcCwohjMzsKZ3FGN+Bq/e+c0xHGfiWtEznkg6SNnW/dIxjs0YJ2AMnM3tJSFPgHkKY17gBcAQ+zOw5A3aSbsCkdW0NnNOZY2rstpcInJ3cS/SzwGdqtSzLmdusquqtIXWsehVF8QpcJK1qmxt/TMv6wjE/z0leP27i8Ag8inT/axxtAQ+9o/zn9QD3JOiyTjnQEQAAAABJRU5ErkJggg==&labelColor=1a1a1a&style=flat)](https://pabloagn.com/technologies/) [![website article](https://img.shields.io/static/v1?label=Website&message=Post%20Link&color=e60048&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAACXBIWXMAAAsTAAALEwEAmpwYAAAB+ElEQVR4nO2VOYgUURCGR/BAI4MN1EwjI89EMDYQvNBNNNlcE0VBUdlUUSMjj2BF2UDRePDAwGzNF2GNPIYd8Hjv/6YnEHSf/FIDPTJiu4nJFBTd1Kv6/nrVBd1q/S8DJiU9AmaBm5LOSjoATPwDY0LSQUnnzDArmJOjkqclvQceSHohaR6oJC1JeiPprqT9pZSVg5pSyirH4sw5S1EzbwZwP5jTIwWBdj1meEppZ6/XOyXpCdCX9Am4Fv45Yo+Bk1VV7ag3FNz2kKC7yznvHiX4u3U6nXU55xPAW7vfHfvLmNtmW8NaFux67k0Ea03esTfJJQTj23bHgiNtPNK6jZem3Wpg46Wp23hp2q0GNl6axksjaRGYkXRF0mnHq6ra2HSk/X5/k6RDks6YEazFPwnuBS5KuirptqTnkj4CJZ4zwNFSytqBoP/2wDHgXi33A/BM0i2zzDR7SBC4LGlPr9fb5huVUlYMus45b5E0FYJfgQS8C8/Al7jJVEpp86DODLPMNDs0up7xXBQZVKLLb8CCpIfA+ZzzvpTS+lLKGuAI8DT8cClltc+c49yoWQjGL140ao25oW8QXW1IKe3KOR8Hbkh66ZtI+i7plaG+iR244JjP3HDkXnetGWbVp9XYopHtHgvwWtIPu9+BSx7bssBNDdhqX07xT/Jbz1SBBDGHAAAAAElFTkSuQmCC&labelColor=1a1a1a&style=flat)](https://pabloagn.com/guided-projects/exploratory-data-analysis-pt-2/)

In the [last part](https://pabloagn.com/guided-projects/exploratory-data-analysis-pt-1/) of this 3-segment [Guided Project](https://pabloagn.com/guided-projects) we introduced the concept of **Exploratory Data Analysis** (*EDA*). We also stated a simple business case requested by our client, an insurance company, and proceeded to analyze a data set which was provided to us. We made some initial data exploration and chose a set of risk factors which could be potentially used to predict the severity of a given patient's Lung Cancer condition.

In this section, we will go over 12 different classification algorithms. We will start by preparing our data. We will then discuss in a very general way, the underlying theory behind each model and their assumptions. We will finally implement each method step-by-step and make a performance comparison.

We'll be using Python scripts which can be found in the [Guided Project Repo](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis).

The generated plots and test results from last segment can also be found in the [plots](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis/plots) and [outputs](https://github.com/pabloagn/guided-projects/tree/master/data-science/exploratory-data-analysis/outputs) folder respectively.

---

## Table of Contents
- [Classification model design](#classification-model-design)
	- [Selecting our methods](#1-selecting-our-methods)
	- Creating a Virtual Environment
	- [Preparing our environment](#2-preparing-our-environment)
	- A word on model assumptions
	- Multinomial Logistic Regression
	- Decision Tree
	- Random Forest
	- Nonlinear Support Vector Machine
	- K-Nearest Neighbors
	- K-Means Clustering
	- Gaussian Naïve Bayes
	- Bernoulli Naïve Bayes
	- Stochastic Gradient Descent
	- Gradient Boosting
	- Extreme Gradient Boosting
	- Deep Neural Networks
	- [Method comparison](#12-method-comparison)
- [Conclusions](#conclusions)
- [References](#references)

---

## Classification model design
**Classification models** are a subset of **supervised machine learning algorithms**. A typical classification model reads an input and tries to classify it based on some predefined properties. A very simple example would be the classification of a mail *containing spam* vs one *without spam*.

The other type of supervised algorithms, perhaps more familiar, are **regression models**. These differ in that they don't classify our inputs into categories, but predict continuous variables. A common example would be predicting the stock market behavior for a given asset.

### 1. Selecting our methods
There are multiple supervised models we can implement to try to predict the severity of Lung Cancer for a given patient. It's always a good idea to test at least a set of different models and compare their accuracy. Since we have categorical ordinal variables, we will test different classification algorithms.

It's also important to consider that not every classification algorithm is appropriate for every classification problem. Each model is based on assumptions that may render it unusable for certain applications.

In this example, we will be working with 12 classification models, which we'll explain in more detail further on:
- Multinomial Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- K-Means Clustering
- Gaussian Naïve Bayes
- Bernoulli Naïve Bayes
- Stochastic Gradient Descent
- Gradient Boosting
- Extreme Gradient Boosting
- Deep Neural Networks

### 2. Creating a Virtual Environment
Before anything else, we need to check our current Python version. This is important because although we'll not be using `tensorflow` directly, we will require it for our Deep Neural Network model using `keras`, and `tensorflow` currently supports **Python versions 3.7 - 3.10*:

##### **Code**
```
import sys
sys.version
```

##### **Output**
```
'3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)]'
```

We can consult the `tensorflow` installation requirements for each operating system [here](https://www.tensorflow.org/install/pip).

If we have a Python version within the range above, we'll be fine and can skip to the module installation part. Otherwise, we have two options:
- Install an older version of Python user-wide or system-wide, and use if as our default interpreter.
- Create a new virtual environment containing a downgraded Python version.

The second option is always best practice, simply because one other program we wrote might be using a newer Python version, and if we replace our current Python version with an older one, we could break any program we wrote using more recent versions. Virtual environments handle these types of conflicts for us; we can have multiple Python installations and selectively choose which environment to work with depending on each case.

Since we require a different Python version than the one we have, we will first download an install our target version by heading to the [Python Releases for Windows](https://www.python.org/downloads/windows/) site.

We will then select the version that we want to download. For this case we will use [Python 3.10.0 - Oct. 4, 2021](https://www.python.org/downloads/release/python-3100/) by getting the corresponding 64-bit Windows installer. Upon download, we will execute the installer and wait for it to conclude. A new Python version will be installed on our system.

Since we installed it user-wide, the executable will be found on `C:/Users/our_username/AppData/Local/Programs/Python`. We must remember this path since we will use it to point the Python version upon our `venv` creation.

We will then create a new virtual environment dedicated for this project. For this, we will need to first `cd` into our project directory:

##### **Code**
```PowerShell
cd 'C:/Users/our_username/exploratory-data-analysis'
```

We will then create the environment using the built-in `venv` package. We can provide whichever name we like. Since we don't have Python 3.10 specified in `PATH`, we will need to refer to it by specifying the full absolute path.

##### **Code**
```PowerShell
C:/Users/our_username/AppData/Local/Programs/Python/Python310/python.exe -m venv 'eda_venv'
```

We will see that a new folder was created on our working directory:

##### **Code**
```PowerShell
ls
```

##### **Output**
```
eda_venv
outputs
plots
cancer patient data sets.csv
exploratory-data-analysis-1.py
exploratory-data-analysis-2.py
```

We can then activate our environment:

##### **Code**
```PowerShell
cd eda_venv\Scripts

.\Activate.ps1
```

We must keep in mind that this `Activate.ps1` is intended to be run by Microsoft PowerShell. If we're running a different shell, we must check which `activate` version to use. For `cmd`, the `activate.bat` file should be executed instead.

We are now inside our virtual environment using Python 3.10. To confirm, we can look at the left of our command prompt, and it should display `eda_venv`.

To start using our environment in our IDE, there's one additional step we must perform; this heavily depends on which IDE we're using, but normally we'll have to point it to our new interpreter (`eda_venv/Scripts/python.exe`) by specifying its path on our preferences menu.

- On Spyder:
	- We can head to *Tools*, *Preferences*, *Python Interpreter*.
	- We can then input the interpreter's path.

- On VS Code:
	- We can open the command palette by pressing <kbd>F1</kbd>.
	- We can then search for *Python: Select Interpreter*.
	- We can input our interpreter's path.

If we're using a version control system such as [GitHub](https://pabloagn.com/technologies/github/), the best practice is to add our `eda_venv` folder to our `.gitignore` file. We can manage the required dependencies for our project by using a `requirements.txt`. For this, we will create a new `requirements.txt` file and place it in our folder project:

##### **Code**
```PowerShell
cd exporatory-data-analysis

New-Item requirements.txt
```

We will then include the following and save it:

##### **Code**
```txt
matplotlib
seaborn
numpy
pandas
scipy
scikit-learn
keras
xgboost
tensorflow==2.10
xlsxwriter
```

If we're using a Windows machine, we can install `tensorflow r2.10` since this was the last release to support GPU processing on native-Windows. We can also stick with the `tensorflow-cpu` package since our data set is not extense, but `tensorflow` really leverages GPU processing power to perform faster, specially in deep learning models. For this segment, we will be using the GPU powered `tensorflow` package, hence the version definition on our `requirements.txt` file.

We will also need to install the NVIDIA CUDA Toolkit & the CUDA Deep Neural Network (*cuDNN*) library if we wish to enable GPU processing. We can head to the [CUDA Toolkit Downloads page](https://developer.nvidia.com/cuda-downloads) and get the version for our case (*it's important to read all CUDA requirements, i.e. Visual Studio is required for it to work properly. Also, `tensorflow` requires a specific CUDA version*). For cuDNN, we can head to the [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) page (*we will have to create an NVIDIA developer account for this one*).

### 3. Preparing our environment
Now that we have our environment ready, we can install all our packages using the `requirements.txt` file we just generated:

##### **Code**
```PowerShell
cd exploratory-data-analysis

pip install -r requirements.txt
```

And that's it, we have every package we need installed on our virtual environment and ready to be imported.

We can then import the required modules:

##### **Code**
```Python
# Data manipulation modules
import pandas as pd
import numpy as np

# Plotting modules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

# Preprocessing modules
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

# Evaluation & performance modules
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score, mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error, log_loss
from scipy import stats
from scipy.stats import norm, skew

# Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import keras
import keras.activations, keras.losses, keras.metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Utility modules
import warnings
import shutil
```

We can suppress unnecessary warnings and define plot parameters:

##### **Code**
```Python
# Supress verbose warnings
warnings.filterwarnings("ignore")

# Define plot parameters
# Before anything else, delete the Matplotlib
# font cache directory if it exists, to ensure
# custom font propper loading
try:
    shutil.rmtree(matplotlib.get_cachedir())
except FileNotFoundError:
    pass

# Define main color as hex
color_main = '#1a1a1a'

# Define title & label padding
text_padding = 18

# Define font sizes
title_font_size = 17
label_font_size = 14

# Define rc params
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Lora']
```

As we have multiple models, it will be best to build a dictionary with each model name as `Key`, and each model as `Value`. We will also define our model parameters inside each model, so we don't have to define them as additional variables in our workspace:

##### **Code**
```Python
model_dictionary = {
    'Multinomial Logistic Regressor': LogisticRegression(random_state=42, max_iter=100000),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
	'K-Nearest Neighbors Classifier' : KNeighborsClassifier(),
	'K-Means Clustering' : KMeans(),
    'Gaussian Naive Bayes Classifier': GaussianNB(),
    'Bernoulli Naive Bayes Classifier': BernoulliNB(),
    'Stochastic Gradient Descent': SGDClassifier(loss='squared_error', max_iter=10000, random_state=42),
	'Gradient Boosting Classifier': GradientBoostingClassifier(),
	'Extreme Gradient Boosting Classifier' : XGBClassifier(),
	'Deep Neural Network' : Sequential()
}
```

### 4. A word on model assumptions
Assumptions denote the collection of explicitly stated (*or implicit premised*) conventions, choices and other specifications on which any model is based.

Every model is built on top of assumptions. They provide the theoretical foundation for it to exist and be valid, and machine learning models are no exception. That is not to say that every assumption must be rigorously met for a given model to work as expected, but also, we cannot bypass every assumption and expect our model to work as designed.

If we understand the underlying theory behind our model, we can be selective in the assumptions we can live without; we can gain knowledge on the implications of bypassing a certain assumption, and can thus make a supported decision on which model to use. It's a matter of balance, and finding out what's good for our specific application.

We will discuss in a very general way, the underlying theory behind each model. We will also list their assumptions and explain what they signify.

### 5. Multinomial Logistic Regression
Multinomial Logistic Regression is a classification method that generalizes logistic regression to multiclass problems, *i.e. when we have more than two possible discrete outcomes*.

#### 5.1 Mathematical intuition overview
Multinomial Logistic regression uses a linear predictor function $f(k,i)$ to predict the probability that observation $i$ has outcome $k$, of the following form: 

$$f(k,i)=\beta_{0,k}+\beta_{1,kX_1,i}+\beta_{2,kX_2,i}+\cdots+\beta_{M,kX_M,i}$$
Where:
- $\beta_{m,k}$ is the set of regression coefficients.
- $k$ is the outcome.
- $X_i$ is a row vector containing the set of explanatory variables associated with observation $i$.

We can express our predictor function it in a more compact form, since the regression coefficients and explanatory variables are normally grouped into vectors of size $M+1$:

$$f(k,i)=\beta_k+X_i$$

When fitting a multinomial logistic regression model, we have several outcomes ($K$), which means that we can think  of the problem as fitting $K-1$ independent Binary Logit Models. From the Binary Logit Model equation we can express our predictor functions as follows:

$$\ln\frac{Pr(Y_i=1)}{Pr(Y_i=K)}=\beta_1 \cdot X_i$$
$$\ln\frac{Pr(Y_i=2)}{Pr(Y_i=K)}=\beta_2 \cdot X_i$$
$$\cdots$$
$$\ln\frac{Pr(Y_i=K-1)}{Pr(Y_i=K)}=\beta_{K-1} \cdot X_i$$

We can then exponentiate both sides of our equation to get actual probabilities:

$$Pr(Y_i=1)={Pr(Y_i=K)}\cdot e^{\beta_1 \cdot X_i}$$
$$Pr(Y_i=2)={Pr(Y_i=K)}\cdot e^{\beta_2 \cdot X_i}$$
$$\cdots$$
$$Pr(Y_i=K-1)={Pr(Y_i=K)} \cdot e^{\beta_K-1 \cdot X_i}$$

#### 5.2 Assumptions
- Requires the dependent variable to be binary, multinomial or ordinal in nature.
- It has a linear decision surface, meaning it can’t solve non-linear problems.
- Requires very little to no multicollinearity, meaning our independent variables must not be correlated with each other.
- Usually works best with large data sets and also requires sufficient training examples for all the categories to make correct predictions.

#### 5.3 Implementation

### 6. Decision Tree
A Decision Tree is a technique than can be used for both classification and regression problems. In our case, we'll be using a Decision Tree Classifier.

A Decision Tree has two types of nodes:
- **Decision Node:** Are in charge of making decisions, and branch in multiple nodes. 
- **Leaf Node:** Are the outputs of the decision nodes and do not branch further.

A Decision Tree algorithm starts from the root node of the tree containing the full data set. It then compares values of the best attribute using **Attribute Selection Measures (*ASM*)**. It then divides the root node into subsets containing possible values for the best attributes. It then generates a new node, which contains the best attribute. Finally, it recursively makes new decision trees using the subsets of the dataset and continues until a stage is reached where it cannot further classify the nodes. This is where the final node (*leaf node*) is created.

#### 6.1 Mathematical intuition overview
Attribute Selection Measures (*ASM*) determine which attribute to select as decision node and branch further. There are two main ASMs: 

##### 6.1.1 Information Gain
Measures the change in entropy after the segmentation of a dataset based on an attribute occurs:

$$Gain (S,a)= Entropy(S)-\sum_{v \in V(A)}\frac{|S_v|}{|S|} \cdot Entropy(S_v)$$

We can interpret entropy as impurity in a given attribute:

$$Entropy(s)=\sum_{i=1}^{n}-p(c_i)\cdot log_2(p(c_i))$$
Where:
- $S$ is the data set $S$.
- $S_v$ is the dataset $S_v$.
- $\frac{|S_v|}{|S|}$ represents the proportion of the values in $S_v$ to the number of values in dataset, $S$.
- $p(c_i)$ is the probability of class $c_i$ in a node.

The more the entropy removed, the greater the information gain. The higher the information gain, the better the split.

##### 6.1.2 Gini Index
Measures impurity; if all the elements belong to a single class, then it can be called pure. The degree of Gini Index varies between 0 and 1. A Gini Index of 0 denotes that all elements belong to a certain class or there exists only one class (*pure*). A Gini Index of 1 denotes that the elements are randomly distributed across various classes (*impure*).

Gini Index is expressed with the following equation:

$$Gini = 1-\sum_{i=1}^{n}p^2(c_i)$$

Where:
- $p^2(c_i)$ is the squared probability of class $c_i$ in a node.

#### 6.2 Assumptions
- In the beginning, the whole training set is considered as the root.
- Feature values are preferred to be categorical.
- Records are distributed recursively on the basis of attribute values.

#### 6.3 Implementation

### 7. Random Forest
Random Forest is an ensemble learning method for classification, regression and other methods. It works by constructing a multitude of decision trees at training time; the output of the random forest is the class selected by most trees.

#### 7.1 Mathematical intuition overview
The training algorithm for random forests applies a generalization of **bagging**.

Given a training set $X = x_1, \cdots , x_n$ with responses $Y = y_1, \cdots , y_n$ bagging repeatedly (*B times*) selects a random sample with replacement of the training set and fits trees to these samples.

After training, predictions for unseen samples $x'$ can be made by averaging the predictions from all the individual regression trees on $x'$ or by taking the majority vote from the set of trees.

We can also include a measure of the uncertainty of the prediction calculating the standard deviation of the predictions from all the individual regression trees on $x'$.

#### 7.2 Assumptions
- It inherits assumptions from the decision tree model.
- There should be some actual values in the feature variables of the dataset, which will give the classifier a better chance to predict accurate results.
- The predictions from each tree must have very low correlations.

#### 7.3 Implementation

### 8. Nonlinear Support Vector Machine
Support Vector Machines (*SVM*) are a class of supervised models originally developed for linear applications, although a nonlinear implementation using nonlinear Kernels was also developed; the resulting algorithm is similar, except that every dot product is replaced by a nonlinear kernel function.

#### 8.1 Mathematical intuition overview
The SVM model amounts to minimizing an expression of the following form:

$$\left[\frac{1}{n} \sum_{i-1}^{n}max(0,1-y_i \cdot (W^\top x_i-b)) \right]+\lambda||w||^2$$

Where:
- $\sum_{i-1}^{n}max(0,1-y_i \cdot (W^\top x_i-b)$ is the loss function.
- $\lambda||w||^2$ is the regularization.

With the different nonlinear Kernels being:
- Polynomial homogeneous (*when $d=1$, this becomes the linear kernel*): $k(X_i, X_j)=(X_i \cdot X_j)^d$
- Polynomial homogeneous: $k(X_i, X_j)=(X_i \cdot X_j + r)^d$
- Gaussian radial basis function: $k(X_i, X_j)=e^{-\lambda \cdot||X_i-X_j||^2}$, for $\lambda > 0$
- Sigmoid function: $k(X_i, X_j)=tanh(kX_i \cdot X_j+c)$, for some $k>0$ and $c<0$

#### 8.2 Assumptions
There are no particular assumptions for this model. If we scale our variables, it might increase performance, but is not required.

#### 8.3 Implementation

### 9. K-Nearest Neighbors
K-Nearest Neighbors (*KNN*) is a non-parametric, supervised learning classifier which uses proximity to classify and group data points. A class label is assigned on the basis of a majority vote *i.e. the label that is most frequently represented around a given data point is used*. The KNN model chooses $k$ nearest points by calculating distances using different metrics, and finally calculates an average to make a prediction.

#### 9.1 Mathematical intuition overview
There are several distance metrics that can be used:

##### 9.1.1 Euclidean distance
This is the most one and it is limited to real-valued vectors. It measures a straight line between two points:

$$d(x, y)=\sqrt{\sum_{i=1}^{n}(y_i-x_i)^2}$$
##### 9.1.2 Manhattan distance
It is also referred to as taxicab distance or city block distance as it is commonly visualized with a grid:

$$d(x,y)=\sum_{i=1}^{m}|X_i-Y_i|$$

##### 9.1.3 Minkowski distance
This metric is the generalized form of Euclidean and Manhattan distance metrics. Euclidean distance takes $p=2$, while Manhattan distance takes $p=1$ 

$$d(x,y)=\left(\sum_{i=1}^{m}|X_i-Y_i|\right)^\frac{1}{p}$$

##### 9.1.4 Hamming distance
This technique is typically used with Boolean or string vectors. Interestingly, it's also used in **information theory** as a way to measure the distance between two strings of equal length:

$$D_H=\sum_{i=1}^{k}|X_i-Y_i|$$
- If $x=y$, $D=0$,
- If $x \neq y$, $D \neq 1$

#### 9.2 Assumptions
- Items close together in the data set are typically similar

#### 9.3 Implementation

### 10. K-Means Clustering
K-Means Clustering is a classification method originally used in **signal processing**, that aims to partition $n$ observations into $k$ clusters in which each observation belongs to the cluster with the nearest mean.

#### 10.1 Mathematical intuition overview
The K-Means Clustering algorithm aims to minimize an objective function also called the squared error function:

$$J(V)=\sum_{i=l}^{c}\sum_{j=l}^{c_i}(||x_i-v_i||)^2$$
Where:
- $||x_i-v_i||$ is the Euclidean distance between $x_i$ and $v_j$.
- $c_i$ is the number of data points in $ith$ cluster.
- $c$ is the number of cluster centers.

#### 10.2 Assumptions
- The learning algorithm requires apriori specification of the number of cluster centers.
- Points can be measured using Euclidean distances.
- The learning algorithm provides the local optimal of the squared error function.
- Unable to handle noisy data and outliers.
- Applicable only when mean is defined.

#### 10.3 Implementation

### 11. Gaussian Naïve Bayes
Gaussian Naïve Bayes (*GNB*) is a probabilistic machine learning algorithm based on the [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). It is the extension of the Naïve Bayes algorithm, and as its name suggest, it approximates class-conditional distributions as a Gaussian distribution, with a mean $\mu$ and a standard deviation $\sigma$.

#### 11.1 Mathematical intuition overview
We can start from the Bayes' Theorem:

$$P(A|B)=\frac{P(A \cap B)}{P(B)}=\frac{P(A) \cdot P(B|A)}{P(B)}$$
Where:
- $P(A)$ is the probability of $A$ occurring.
- $P(B)$ is the probability of $B$ occurring.
- $P(A|B)$ is the probability of $A$ given $B$.
- $P(B|A)$ is the probability of $B$ given $A$.
- $P(A \cap B)$ is the probability of $A$ and $B$ occurring.

We can then translate the formula above to the Gaussian Naïve Bayes equation:

$$P(X_i|y)=\frac{1}{ \sqrt{2 \pi \sigma_y ^2}}e\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2} \right)$$

We can see that the form of this equation is almost identical to the Gaussian distribution density function. The main difference is that in the first one we're defining our function as a probability function, while in the latter we're defining it as a density function:

$$f(X|\mu,\sigma^2)=\frac{1}{ \sqrt{2 \pi \sigma_y ^2}}e\left(-\frac{(x-\mu)^2}{2\sigma_y^2} \right)$$

#### 11.2 Assumptions
- Features are independent (*hence Naïve*).
- Class-conditional densities are normally distributed.

#### 11.3 Implementation

### 12. Bernoulli Naïve Bayes
Bernoulli Naïve Bayes (*BNB*) is similar to Gaussian Naïve Bayes, in that it also uses Bayes' Theorem as foundation. The difference is that Bernoulli Naïve Bayes approximates class-conditional distributions as a Bernoulli distribution. This fact makes this variation more appropriate for discrete random variables instead of continuous ones.

#### 12.1 Mathematical intuition overview
Since we already went over the Bayes' Theorem, we can start by defining the Bernoulli distribution function:

$$
p(x) =
P[X=x] =
\begin{cases}
p     & \text{if $x = 1$}, \\
q=1 - p & \text{if $x = 0$}.
\end{cases}
$$

From the above, we can then define the Bernoulli Naïve Bayes Classifier:

$$P(x_i|y)=P(i|y)x_i+(1-P(i|y))(1-x_i)$$

#### 12.2 Assumptions
- The attributes are independent of each other and do not affect each other's performance (*hence Naïve*).
- All of the features are given equal importance.

#### 12.3 Implementation

### 13. Stochastic Gradient Descent
Stochastic Gradient Descent (*SGD*) is an optimization method. It can be used in conjunction with other machine learning algorithms.

In general, gradient descent is used to minimize a cost function. There are three main types:

- Batch gradient descent
- Mini-batch gradient descent
- Stochastic gradient descent

Stochastic Gradient Descent computes the gradient by calculating the derivative of the loss of a single random data point rather than all of the data points (*hence the name, stochastic*). It then finds a minimum by taking steps. What makes it different over other optimization methods is it's efficiency, *i.e. it only uses one single random point to calculate the derivative*.

Stochastic Gradient Descent Classifier is a linear classification method with SGD training.

#### 13.1 Mathematical intuition overview
The SGD gradient function can be expressed as follows:

$$\theta^{(t+1)} = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i)}; y^{(i)})$$
Where:
- $x^{(i)}$ is a given training example.
- $y^{(i)}$ is a given label.
- $\nabla_\theta J( \theta)$ is the true gradient of  $J( \theta)$
- $\theta^{(t+1)}$ is the approximation of the true gradient $\nabla_\theta J( \theta)$ at time $t+1$ by a gradient at a single sample.
- $\theta$ is the position of the previous step.

As the algorithm sweeps through the training set, it performs the above update for each training sample. Several passes can be made over the training set until the algorithm converges.

#### 13.2 Assumptions
- The errors at each point in the parameter space are additive
- The expected value of the observation picked at random is a subgradient of the function at point $\theta$.

#### 13.3 Implementation

### 14. Gradient Boosting
Gradient Boosting (*GBM*) is a machine learning technique used in regression and classification tasks to create a stronger model by using an ensemble of weaker models. The objective of Gradient Boosting classifiers is to minimize the loss, or the difference between the actual class value of the training example and the predicted class value. As with other classifiers, GBM depends on a loss function, which can be customized to improve performance.

Gradient Boosting Classifiers consist of three main parts:
- The weak model, usually a Decision Tree
- The additive component
- A loss function that is to be optimized

The main problem with Gradient Boosting is the potential of overfitting the model. We know that perfect training scores will lead to this phenomenon. This can be overcome by setting different regularization methods such as tree constraints, shrinkage and penalized learning.

#### 14.1 Mathematical intuition overview
We can generalize a Gradient-Boosted Decision Tree model.

We can initialize our model with a constant loss function:

$$F_0(x)=\text{arg min}\sum_{i=1}^{n}L(y_i,\gamma)$$

We can then compute the residuals:

$$r_{im}=-\left[ \frac{\delta L(y_i, F(x_i))}{\delta F(x_i)} \right]_{F(x)=F_{m-1}(x)}\hspace{0.1cm},for\hspace{0.1cm}i=1,\cdots ,n$$

We can then train our Decision Tree with features $x$ against $r$ and create terminal node regressions $R_{jm}$.

Next, we can compute a $\gamma_{jm}$ which minimizes our loss function on each terminal node:

$$\gamma_{jm}=\text{arg min}\sum_{x_i \in R_{jm}}^{n}L(y_i,F_{m-1}(x_i) + \gamma),\hspace{0.1cm}for\hspace{0.1cm} j=1, \cdots J_m$$

Finally, we can recompute our model with our new $\gamma_{jm}$:

$$F_m(x)=F_{m-1}(x)+v\sum_{j=1}^{J_m}\gamma_{jm1}(x \in R_{jm})$$

Where:
- $r_{im}$ is the residual or gradient of our loss function.
- $F_o$ is our first iteration.
- $F_m$ is the updated prediction.
- $F_{m-1}(x)$ is the previous prediction.
- $v$ is the learning rate between 0 and 1.
- $\gamma_{jm1}$ is the value which minimizes the loss function on each terminal node.
- $R_{jm}$ is the terminal node.

#### 14.2 Assumptions
- The sum of its residuals is 0, *i.e. the residuals should be spread randomly around zero*.

#### 14.3 Implementation

### 15. Extreme Gradient Boosting
Extreme Gradient Boosting (*XGBoost*) is a more regularized form of the previous Gradient Boosting technique. This means that it controls overfitting better, resulting in better performance; as opposed to GBM, XGBoost uses advanced regularization (*L1 & L2*), which improves model generalization capabilities. It also has faster training capabilities and can be parallelized across clusters, reducing training times.

Some other differences of XGBoost over GBM are:
- The use of sparse matrices with sparsity aware algorithms.
- Improved data structures for better processor cache utilization which makes it faster.

We will skip the mathematical intuition for XGBoost since it's extense and similar to it's cousin GBM's.

#### 15.2 Assumptions
- Encoded integer values for each input variable have an ordinal relationship.
- The data may not be complete (*can handle sparsity*)

#### 15.3 Implementation

### 16. Deep Neural Networks
A Deep Neural Network is simply a Neural Network containing at least two interconnected layers of neurons. Its functioning as well as the theory behind them are somewhat different to what we've seen so far. Also, they belong to a different branch of Artificial Intelligence called [Deep Learning](https://www.mathworks.com/discovery/deep-learning.html), which is itself a subgroup of [Neural Networks](https://www.ibm.com/topics/neural-networks). The model that would assimilate more (*in a sense*) would be Decision Trees, although even they process data differently.

Neural Networks were created based on how actual neurons work (*in a very general way*); they are comprised of node layers containing an input layer, one or more hidden layers, and an output layer. Each node connects to another and has an associated weight and threshold. These parameters are what define the signal intensity from one neuron to another; if the output of a given individual node is above the specified threshold value, that node is activated, sending a signal to the next layer of the network, else, the signal doesn't pass through.

Although Deep Neural Networks can achieve complex classification tasks, there are some major disadvantages:

- It takes time and domain-knowledge to fine-tune a Neural Network.
- They're sensitive to data inputs.
- They are computationally expensive, making them a challenge to deploy on a production environment.
- Their hidden layers work as black boxes making them hard to understand or debug.
- Most of the times, they require more data to return accurate results.
- They rely more on training data, potentially leading to overfitting.

Many times, a simpler alternative such as Decision Tree Classifier gives better accuracy without all the disadvantages above.

Apart from all the points mentioned, there are also major advantages:

- They are able to perform unsupervised learning.
- They have good fault tolerance, meaning the output is not affected by the corruption of one or more than one cell.
- They have distributed memory capabilities.

#### 16.1 Mathematical intuition overview
As we have mentioned, a Neural Network works by propagating signals depending on the weight and threshold of each neuron.

The most basic Neural Network is called *perceptron*, and consists of $n$ number of inputs, one neuron, and one output.

A perceptron's forward propagation starts by weighting each input and adding all the multiplied values. Weights decide how much influence the given input will have on the neuron’s output:

$$\sum=(x_1w_1)+(x_2w_2)+(x_3w_3)+ \cdots +(x_nw_n)=x \cdot w$$

Where:
- $x$ is a vector of inputs.
- $w$ is a vector of weights.
- $x \cdot w$ is the dot product between $x$ and $w$.

Then, a bias is added to the summation calculated before:

$$z=x \cdot w + b$$

Where:
- $b$ is the bias

Finally, we pass $z$ to a non-linear activation function. Perceptrons have binary step functions as their activation functions. This is the most simple type of function; it produces a binary output:

$$
f(x) =
\begin{cases}
0     & \text{if $x < 0$}, \\
1     & \text{if $x \geq 0$}.
\end{cases}
$$

A perceptron is the simplest case, and of course, the more layers we have, the more complex the mathematical derivation gets. Also, there are more complex and appropriate activation functions available, since the binary activation functions presents important disadvantages.

#### 16.2 Assumptions
- Artificial Neurons are arranged in layers, which are sequentially arranged.
- Neurons within the same layer do not interact or communicate to each other.
- All inputs enter into the network through the input layer and pass through the output layer.
- All hidden layers at same level should have same activation function.
- Artificial neuron at consecutive layers are densely connected.
- Every inter-connected neural network has it’s own weight and biased associated with it.

#### 16.3 Implementation

### 17. Method comparison


## Conclusions


## References
- [IBM, KNN](https://www.ibm.com/topics/knn)
- [OpenGenus, Gaussian Naïve Bayes](https://iq.opengenus.org/gaussian-naive-bayes/)
- [OpenGenus, Bernoulli Naïve Bayes](https://iq.opengenus.org/bernoulli-naive-bayes/)
- [Michael Fuchs, Introduction to SGD Classifier](https://michael-fuchs-python.netlify.app/2019/11/11/introduction-to-sgd-classifier/)
- [Jerry Friedman, Greedy Function Approximation: A Gradient Boosting Machine](https://jerryfriedman.su.domains/ftp/trebst.pdf)

## Pendings
- https://www.educative.io/answers/implement-neural-network-for-classification-using-scikit-learn

## Model Implementation
- https://www.kaggle.com/code/ayushs9020/lung-cancer-prediction-99-98
- https://www.kaggle.com/code/mdsajidanamifti/lung-cancer-prediction-and-visualization
- https://www.kaggle.com/code/sripadkarthik/lung-cancer-prediction-using-ml-and-dl
- https://www.kaggle.com/code/youssefabdelmottaleb/lungs-cancer-classification-ml-dl#EDA-&-Data-Preprocessing
- https://www.kaggle.com/code/guslovesmath/lung-cancer-prediction-ml#-4---Model-Building-
- https://www.kaggle.com/code/yashrandive11/lungcancerclassifier#Exploratory-Data-Analysis
- https://www.kaggle.com/code/pedrodicati/lung-cancer-with-accuracy-99-2
- https://www.kaggle.com/code/ekasabrol/keras-dl-eda-roc-auc-lung-cancer
- https://www.kaggle.com/code/baturalpsert/deep-learning-lung-cancer#Building-Neural-Network-Model
- https://www.kaggle.com/code/stpeteishii/lung-cancer-predict-visualize-importance
- https://www.kaggle.com/code/ayushb6732/lungs-dataset-gdsc
- https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/

