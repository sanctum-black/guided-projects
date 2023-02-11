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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
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
    'Multinomial Logistic Regressor': LogisticRegression(multi_class='multinomial',
							  solver='lbfgs',
							 random_state=42,
							 max_iter=100000,
							    penalty='l2',
							           C=24),
    'Logistic Regressor' : LogisticRegression(C=24),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
	'K-Nearest Neighbors Classifier' : KNeighborsClassifier(),
	'K-Means Clustering' : KMeans(),
    'Gaussian Naive Bayes Classifier': GaussianNB(),
    'Bernoulli Naive Bayes Classifier': BernoulliNB(),
    'Stochastic Gradient Descent': SGDClassifier(loss='squared_error',
                                                 max_iter=10000,
                                                 random_state=42),
	'Gradient Boosting Classifier': GradientBoostingClassifier(),
	'Extreme Gradient Boosting Classifier' : XGBClassifier(),
	'Deep Neural Network' : Sequential()
}
```

We can then define a dictionary which will contain all the preprocessing functions that we will need:

##### **Code**
```Python
preprocessing_dictionary = {'Right Skew Gaussian' : FunctionTransformer(func = np.square),
                            'Left Skew Gaussian' : FunctionTransformer(func = np.log1p),
                            'Standard Scaler' : StandardScaler()
}
```

We can also define a third dictionary which contain all the evaluation techniques:

##### **Code**
```Python
# Define evaluation techniques
evaluation_dictionary = {'KFold' : RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
}
```

We can then import our data set, and do some preprocessing:

##### **Code**
```Python
# Read the data set
df = pd.read_csv('cancer patient data sets.csv')

# Remove index column
df.drop(columns = "index", inplace = True)

# Map Level to numeric values
illness_level_dict = {'Low' : 1,
                      'Medium' : 2,
                      'High': 3}

df['Level'] = df['Level'].map(illness_level_dict)

# Remove columns that we will not study
remove_cols = ['Patient Id',
               'Gender',
               'Age',
               'Chest Pain',
               'Coughing of Blood',
               'Fatigue',
               'Weight Loss',
               'Shortness of Breath',
               'Wheezing',
               'Swallowing Difficulty',
               'Clubbing of Finger Nails',
               'Frequent Cold',
               'Dry Cough',
               'Snoring']

df = df.drop(columns = remove_cols)

print(df.shape)
print(list(df.columns))
```

We end up with a DataFrame with the following characteristics:

##### **Output**
```
(1000, 26)

['Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Level']
```

If we recall from last section, these are the potential risk factors that our client is looking to study. We had to remove all the other symptomatic characteristics as our client is not interested on these.

We will now define a simple function that will help us split our data into train and test sets:

##### **Code**
```Python
def sep(dataframe):
    '''
    Parameters
    ----------
    dataframe : DataFrame
        Contains our data as a DataFrame object.

    Returns
    -------
    x : DataFrame
        Contains our features.
    y : DataFrame
        Contains our labels.
    '''
    target = ["Level"]
    x = dataframe.drop(target , axis = 1)
    y = dataframe[target]
    
    return x, y
```

We will now define three functions that will help us with the results generation:
- `cm_plot` will plot a confusion matrix for each method. Confusion matrixes are a special kind of contingency table, with two dimensions (*actual* and *predicted*). The idea behind the confusion matrix, is to get a quick graphical grasp of how our model performed in predicting compared to the test data. It is a widely used method and a very simple one to implement and explain to a non-technical audience.
- `model_score` will calculate the model score as the $R^2$ coefficient.
- `classification_rep` will calculate $\text{precision}$, $\text{recall}$, $\text{f1-score}$ and $\text{support}$ for each label, and return it as a DataFrame object.

##### **Code**
```Python
# Define Confusion Matrix Function
def cm_plot(model_name, model, test_y, predicted_y):
    '''
    Parameters
    ----------
    model_name : Str
        Contains the used model name.
    model : sklearn or keras model object
        Contains a model object depending on the model used.
    test_y : DataFrame
        Contains the non-scaled test values for our data set.
    predicted_y : Array
        Contains the predicted values for a given method.

    Returns
    -------
    None.
    '''
    cm = confusion_matrix(test_y, predicted_y)
    plt.figure(f'{model_name}_confusion_matrix')
    sn.heatmap(cm, annot=True, linewidth=0.7, cmap="rocket")
    plt.title(f'{model_name} Confusion Matrix\n')
    plt.xlabel('y Predicted')
    plt.ylabel('y Test')
    plt.savefig('plots/' + f'{model_name}_confusion_matrix.png', format = 'png', dpi = 300, transparent = True)
    plt.close()
    return None

# Define model score
def model_score(model, test_x, test_y):
    '''
    Parameters
    ----------
    model : sklearn or keras model object
        Contains a model object depending on the model used.
    test_x : Array
        Contains the transformed / scaled test values for the features.
    test_y : DataFrame
        Contains the un-scaled / un-transformed test values for the labels.

    Returns
    -------
    sc : Float
        Contains the score model.
    '''
    sc = model.score(test_x, test_y)

    return sc

# Define Classification Report Function
def classification_rep(test_y, predicted_y):
    '''
    Parameters
    ----------
    test_y : DataFrame
        Contains the non-scaled test values for our data set.
    predicted_y : Array
        Contains the predicted values for a given method.

    Returns
    -------
    cr : DataFrame
        Contains a report showing the main classification metrics.
    '''
    cr = classification_report(test_y, predicted_y, output_dict=True)
    cr = pd.DataFrame(cr).transpose()
    
    return cr
```

We will now transform our data in order to make it usable for each model:

##### **Code**
```Python
# For Normal Distribution Methods, we can approximate our data set to
# a normal distribution
right_skew = []
left_skew = []
for i in df_x.columns:
    if df_x[i].skew() > 0:
        right_skew.append(i)
    else:
        left_skew.append(i)

right_skew_transformed = preprocessing_dictionary['Right Skew Gaussian'].fit_transform(df_x[right_skew])
left_skew_transformed = preprocessing_dictionary['Left Skew Gaussian'].fit_transform(df_x[left_skew])

df_gaussian = pd.concat([right_skew_transformed,
                         left_skew_transformed ,
                         df_y] ,
                         axis = 1,
                         join = "inner")

# We can divide into train & text, x & y
train_G, test_G = train_test_split(df_gaussian, test_size=0.2)
train_Gx, train_Gy = sep(train_G)
test_Gx, test_Gy = sep(test_G)

# For other methods, we can scale using Standard Scaler
train, test = train_test_split(df, test_size=0.2)
train_x, train_Sy = sep(train)
test_x, test_Sy = sep(test)

train_Sx = preprocessing_dictionary['Standard Scaler'].fit_transform(train_x)
test_Sx = preprocessing_dictionary['Standard Scaler'].transform(test_x)
```

Now that we have our transformed sets, we can start talking about the selected models. For each case, we will briefly describe what the model is about, it's general mathematical intuition and it's assumptions.

We need to consider that the mathematical background provided in this segment is by any means a rigorous derivation. We could spend an entire series talking about one model's mathematical background. Instead, we will simply review the main mathematical formulae involved in each model.

### 4. A word on model assumptions
Assumptions denote the collection of explicitly stated (*or implicit premised*) conventions, choices and other specifications on which any model is based.

Every model is built on top of assumptions. They provide the theoretical foundation for it to exist and be valid, and machine learning models are no exception. That is not to say that every assumption must be rigorously met for a given model to work as expected, but also, we cannot bypass every assumption and expect our model to work as designed.

If we understand the underlying theory behind our model, we can be selective in the assumptions we can live without; we can gain knowledge on the implications of bypassing a certain assumption, and can thus make a supported decision on which model to use. It's a matter of balance, and finding out what's good for our case.

### 5. Multinomial Logistic Regression
Multinomial Logistic Regression is a classification method that generalizes logistic regression to multiclass problems, *i.e. when we have more than two possible discrete outcomes*.

Logistic Regression, or Logit Model, contrary to what it's name may suggest, is not a regression model but a **parametric classification** one. In reality, this model is very similar to [Linear Regression](https://www.ibm.com/topics/linear-regression); the main difference between the two, is that in Logistic regression we don’t fit a straight line to our data. Instead, we fit an $S$ shaped curve, called **Sigmoid**, to our observations.

#### 5.1 Mathematical intuition overview
**Logistic Regression** fits data to a $\text{Sigmoid}$ function:

$$\text{sigmoid}(x)=\frac{1}{1+e^{-x}}$$

It first calculates a weighted sum of inputs:

$$x=\Theta \cdot feature + b$$
It then calculates the probability of the weighted feature belonging to a given group:

$$P(x)=\frac{1}{1+e^{-x}}$$
Weights are calculated using different optimization models, such as **Gradient Descent** or **Maximum Likelihood**.

**Multinomial Logistic Regression** uses a linear predictor function $f(k,i)$ to predict the probability that observation $i$ has outcome $k$, of the following form: 

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
We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Multinomial Logistic Regressor'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_MLogReg = model_dictionary['Multinomial Logistic Regressor'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Multinomial Logistic Regressor',
        model_dictionary['Multinomial Logistic Regressor'],
        test_Sy,
        y_predicted_MLogReg)

# Define model score
score_MLogReg = model_score(model_dictionary['Multinomial Logistic Regressor'],
                            test_Sx,
                            test_Sy)

# Define Classification Report Function
report_MLogReg = classification_rep(test_Sy,
                                    y_predicted_MLogReg)
                                    
print(score_MLogReg)
```

If we take a look at our results, we can see that it predicted with a **91.5%** accuracy:

##### Output
```
0.915
```

Not to worry, we will explore the results in more detail in the [Method Comparison](#17-method-comparison) section.

We can now use a Binomial Logistic Regression model and see what we get:

##### **Code**
```Python
# Train model
model_dictionary['Logistic Regressor'].fit(train_Sx, train_Sy)

# Predict
y_predicted_BLogReg = model_dictionary['Logistic Regressor'].predict(test_Sx)

# Evaluate the model and collect the scores
cm_plot('Logistic Regressor',
        model_dictionary['Logistic Regressor'],
        test_Sy,
        y_predicted_BLogReg)

# Define model score
score_BLogReg = model_score(model_dictionary['Logistic Regressor'],
                            test_Sx,
                            test_Sy)

# Define Classification Report Function
report_BLogReg = classification_rep(test_Sy,
                                    y_predicted_BLogReg)

print(score_BLogReg)
```

If we take a look at our results, we can see that it predicted with a **91.5%** accuracy. Same as its multinomial cousin:

##### **Output**
```
0.915
```

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
We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Decision Tree Classifier'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_DecTree = model_dictionary['Decision Tree Classifier'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Decision Tree Classifier',
        model_dictionary['Decision Tree Classifier'],
        test_Sy,
        y_predicted_DecTree)

# Define model score
score_DecTree = model_score(model_dictionary['Decision Tree Classifier'],
                            test_Sx,
                            test_Sy)

# Define Classification Report Function
report_DecTree = classification_rep(test_Sy,
                                    y_predicted_DecTree)

print(score_DecTree)
```

If we take a look at our results, we can see that it predicted with a **100%** accuracy:

##### Output
```
1.0
```

The interesting thing about Decision Trees, is that we can actually visualize them using multiple methods,

We can display a simple text representation:

##### **Code**
```Python
# Text Representation
DecTree_text_rep = tree.export_text(model_dictionary['Decision Tree Classifier'])

print(DecTree_text_rep)
```

##### **Output**
```
|--- feature_7 <= 0.99
| |--- feature_2 <= -1.29
| | |--- feature_6 <= 0.50
| | | |--- class: 1
| | |--- feature_6 > 0.50
| | | |--- class: 3
| |--- feature_2 > -1.29
| | |--- feature_9 <= 1.03
| | | |--- feature_1 <= -0.00
| | | | |--- feature_7 <= -0.89
| | | | | |--- feature_1 <= -1.16
| | | | | | |--- feature_3 <= -0.63
| | | | | | | |--- class: 2
| | | | | | |--- feature_3 > -0.63
| | | | | | | |--- class: 1
| | | | | |--- feature_1 > -1.16
| | | | | | |--- class: 1
| | | | |--- feature_7 > -0.89
| | | | | |--- feature_6 <= -0.42
| | | | | | |--- feature_1 <= -0.77
| | | | | | | |--- feature_0 <= -0.14
| | | | | | | | |--- feature_1 <= -1.16
| | | | | | | | | |--- feature_7 <= -0.42
| | | | | | | | | | |--- class: 1
| | | | | | | | | |--- feature_7 > -0.42
| | | | | | | | | | |--- class: 2
| | | | | | | | |--- feature_1 > -1.16
| | | | | | | | | |--- class: 2
| | | | | | | |--- feature_0 > -0.14
| | | | | | | | |--- class: 1
| | | | | | |--- feature_1 > -0.77
| | | | | | | |--- feature_6 <= -0.89
| | | | | | | | |--- class: 1
| | | | | | | |--- feature_6 > -0.89
| | | | | | | | |--- class: 2
| | | | | |--- feature_6 > -0.42
| | | | | | |--- feature_7 <= 0.28
| | | | | | | |--- class: 1
| | | | | | |--- feature_7 > 0.28
| | | | | | | |--- class: 2
| | | |--- feature_1 > -0.00
| | | | |--- feature_5 <= 1.17
| | | | | |--- class: 2
| | | | |--- feature_5 > 1.17
| | | | | |--- class: 1
| | |--- feature_9 > 1.03
| | | |--- class: 3
|--- feature_7 > 0.99
| |--- feature_0 <= -0.63
| | |--- class: 2
| |--- feature_0 > -0.63
| | |--- class: 3
```

We can also plot the tree using `plot_tree`:

##### Code
```Python
# Tree plot using plot_tree
fig = plt.figure('Decision Tree plot_tree')
tree.plot_tree(model_dictionary['Decision Tree Classifier'],
                   feature_names=df_x.columns,
                   class_names=df_y['Level'].astype('str'),
                   filled=True)

plt.title('Decision Tree Plot')
plt.savefig('plots/' + 'decision_tree.png', format = 'png', dpi = 300, transparent = True)
plt.close()
```

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
We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Random Forest Classifier'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_RandomFor = model_dictionary['Random Forest Classifier'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Random Forest Classifier',
        model_dictionary['Random Forest Classifier'],
        test_Sy,
        y_predicted_RandomFor)

# Define model score
score_RandomFor = model_score(model_dictionary['Random Forest Classifier'],
                            test_Sx,
                            test_Sy)

# Define Classification Report Function
report_RandomFor = classification_rep(test_Sy,
                                    y_predicted_RandomFor)

print(score_RandomFor)
```

If we take a look at our results, we can see that it predicted with a **100%** accuracy:

##### Output
```
1.0
```

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
- Gaussian Radial Basis Function (*RBF*): $k(X_i, X_j)=e^\frac{-||X_i-X_j||^2}{2 \sigma^2}$, for $\lambda > 0$
- Sigmoid function: $k(X_i, X_j)=tanh(kX_i \cdot X_j+c)$, for some $k>0$ and $c<0$

#### 8.2 Assumptions
There are no particular assumptions for this model. If we scale our variables, it might increase performance, but is not required.

#### 8.3 Implementation
For this part, we'll be using 3 different approaches; we mentioned that Support Vector Machines are fit for linear applications, although we can use nonlinear Kernels to fit nonlinear data.

There are two particular Kernels we will implement:

- **Polynomial Kernel:** As its name suggests, this Kernel represents the similarity of vectors in a feature space over polynomials of the original variables. We can select the order of the polynomial as a parameter.
- **Radial Basis Function Kernel:** This Kernel is the most generalized form of kernelization and is one of the most widely used in SVM due to its similarity to the Gaussian distribution. 

We can start by fitting our models to our data:

##### **Code**
```Python
# Train models
model_dictionary['Support Vector Classifier'].fit(train_Sx, train_Sy)
model_dictionary['Support Vector Classifier Polynomial Kernel'].fit(train_Sx, train_Sy)
model_dictionary['Support Vector Classifier Radial Kernel'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained models:

##### **Code**
```Python
# Predict
y_predicted_SVM = model_dictionary['Support Vector Classifier'].predict(test_Sx)
y_predicted_SVMp = model_dictionary['Support Vector Classifier Polynomial Kernel'].predict(test_Sx)
y_predicted_SVMr = model_dictionary['Support Vector Classifier Radial Kernel'].predict(test_Sx)
```

We can finally evaluate our models using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Support Vector Classifier',
        model_dictionary['Support Vector Classifier'],
        test_Sy,
        y_predicted_SVM)
        
cm_plot('Support Vector Classifier Polynomial Kernel',
        model_dictionary['Support Vector Classifier Polynomial Kernel'],
        test_Sy,
        y_predicted_SVMp)

cm_plot('Support Vector Classifier Radial Kernel',
        model_dictionary['Support Vector Classifier Radial Kernel'],
        test_Sy,
        y_predicted_SVMr)

# Define model score
score_SVM = model_score(model_dictionary['Support Vector Classifier'],
                        test_Sx,
                        test_Sy)
                                
score_SVMp = model_score(model_dictionary['Support Vector Classifier Polynomial Kernel'], test_Sx, test_Sy)

score_SVMr = model_score(model_dictionary['Support Vector Classifier Radial Kernel'], test_Sx, test_Sy)

# Define Classification Report Function
report_SVM = classification_rep(test_Sy,
                                y_predicted_SVM)
                                
report_SVMp = classification_rep(test_Sy,
                                y_predicted_SVMp)

report_SVMr = classification_rep(test_Sy,
                                y_predicted_SVMr)

print(score_SVM)
print(score_SVMp)
print(score_SVMr)
```

If we take a look at our results, we can see that we get the following accuracies:
- **Linear SVM:** 89.5%
- **Polynomial SVM, 8th degree:** 100%
- **Radial Kernel:** 100%

##### Output
```
0.895
1.0
1.0
```

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
We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['K-Nearest Neighbors Classifier'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_KNN = model_dictionary['K-Nearest Neighbors Classifier'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('K-Nearest Neighbors Classifier',
        model_dictionary['K-Nearest Neighbors Classifier'],
        test_Sy,
        y_predicted_KNN)

# Define model score
score_KNN = model_score(model_dictionary['K-Nearest Neighbors Classifier'],
                        test_Sx,
                        test_Sy)

# Define Classification Report Function
report_KNN = classification_rep(test_Sy,
                                y_predicted_KNN)

print(score_KNN)
```

If we take a look at our results, we can see that it predicted with an **100%** accuracy:

##### Output
```
1.0
```

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
Since we are using the Gaussian variant of the model, we will use the normally-approximated values we generated earlier. We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Gaussian Naive Bayes Classifier'].fit(train_Gx, train_Gy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_GNB = model_dictionary['Gaussian Naive Bayes Classifier'].predict(test_Gx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Gaussian Naive Bayes Classifier',
        model_dictionary['Gaussian Naive Bayes Classifier'],
        test_Gy,
        y_predicted_GNB)

# Define model score
score_GNB = model_score(model_dictionary['Gaussian Naive Bayes Classifier'],
                        test_Gx,
                        test_Gy)

# Define Classification Report Function
report_GNB = classification_rep(test_Gy,
                                y_predicted_GNB)

print(score_GNB)
```

If we take a look at our results, we can see that it predicted with a **71.5%** accuracy. This is the lowest score we've gotten so far:

##### Output
```
0.715
```

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
We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Bernoulli Naive Bayes Classifier'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_BNB = model_dictionary['Bernoulli Naive Bayes Classifier'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Bernoulli Naive Bayes Classifier',
        model_dictionary['Bernoulli Naive Bayes Classifier'],
        test_Sy,
        y_predicted_BNB)

# Define model score
score_BNB = model_score(model_dictionary['Bernoulli Naive Bayes Classifier'],
                        test_Sx,
                        test_Sy)

# Define Classification Report Function
report_BNB = classification_rep(test_Sy,
                                y_predicted_BNB)

print(score_BNB)
```

If we take a look at our results, we can see that it predicted with a **77.5%** accuracy:

##### Output
```
0.775
```

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
For this example, we'll use a Logistic Regressor with a SGD training. We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Stochastic Gradient Descent'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_SGD = model_dictionary['Stochastic Gradient Descent'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Stochastic Gradient Descent',
        model_dictionary['Stochastic Gradient Descent'],
        test_Sy,
        y_predicted_SGD)

# Define model score
score_SGD = model_score(model_dictionary['Stochastic Gradient Descent'],
                        test_Sx,
                        test_Sy)

# Define Classification Report Function
report_SGD = classification_rep(test_Sy,
                                y_predicted_SGD)

print(score_SGD)
```

If we take a look at our results, we can see that it predicted with a **90.5%** accuracy:

##### Output
```
0.905
```

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
For this example, we'll use a Gradient Boosting Classifier. We will leave parameters as default (*100 estimators*), although these can be fine-tuned. We can start by fitting our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Gradient Boosting Classifier'].fit(train_Sx, train_Sy)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_GBC = model_dictionary['Gradient Boosting Classifier'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Gradient Boosting Classifier',
        model_dictionary['Gradient Boosting Classifier'],
        test_Sy,
        y_predicted_GBC)

# Define model score
score_GBC = model_score(model_dictionary['Gradient Boosting Classifier'],
                        test_Sx,
                        test_Sy)

# Define Classification Report Function
report_GBC = classification_rep(test_Sy,
                                y_predicted_GBC)

print(score_GBC)
```

If we take a look at our results, we can see that it predicted with a **100%** accuracy:

##### Output
```
1.0
```

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
For this implementation, we'll use a different library called `XGBoost`. Apart from the advantages on the mathematical treatment, `XGBoost` is written on C++, making it comparatively faster than other Gradient Boosting libraries. Also, `XGBoost` was specifically designed to support parallelization onto GPUs and computer networks. These make this library extremely powerful when handling extense data sets.

Before we can start, we will need to re-encode our labels, since `XGBoost` requires our values to start from 0 and not 1:

##### **Code**
```Python
# Re-encode labels
train_Sy_XGBC = LabelEncoder().fit_transform(train_Sy)
test_Sy_XGBC = LabelEncoder().fit_transform(test_Sy)
```

We will then fit our model to our data:

##### **Code**
```Python
# Train model
model_dictionary['Extreme Gradient Boosting Classifier'].fit(train_Sx, train_Sy_XGBC)
```

We can then predict some values using our trained model:

##### **Code**
```Python
# Predict
y_predicted_XGBC = model_dictionary['Extreme Gradient Boosting Classifier'].predict(test_Sx)
```

We can finally evaluate our model using the evaluation metrics we defined earlier:

##### **Code**
```Python
# Evaluate the model and collect the scores
cm_plot('Extreme Gradient Boosting Classifier',
        model_dictionary['Extreme Gradient Boosting Classifier'],
        test_Sy_XGBC,
        y_predicted_XGBC)

# Define model score
score_XGBC = model_score(model_dictionary['Extreme Gradient Boosting Classifier'],
                        test_Sx,
                        test_Sy_XGBC)

# Define Classification Report Function
report_XGBC = classification_rep(test_Sy_XGBC,
                                y_predicted_XGBC)

print(score_XGBC)
```

If we take a look at our results, we can see that it predicted with a **100%** accuracy:

##### Output
```
1.0
```

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

The theory behind Deep Neural Networks is extense and complex, so we will not be explaining each step on detail; instead, we will stick with a general description of what is being done. A rigorous & exhaustive explanation of these models can be found in [Philipp Christian Petersen's Neural Network Theory](http://pc-petersen.eu/Neural_Network_Theory.pdf).

#### 16.2 Assumptions
- Artificial Neurons are arranged in layers, which are sequentially arranged.
- Neurons within the same layer do not interact or communicate to each other.
- All inputs enter into the network through the input layer and pass through the output layer.
- All hidden layers at same level should have same activation function.
- Artificial neuron at consecutive layers are densely connected.
- Every inter-connected neural network has it’s own weight and biased associated with it.

#### 16.3 Implementation
Deep Neural Networks require a different treatment than the ones we've already seen. The first thing we'll need to do, is to define which model we're going to use. For this case, a simpler 5-layer Sequential model will suffice.

A Sequential Neural Network passes on the data and flows in sequential order from top to bottom approach till the data reaches at end of the model.

We can start by making defining our model:

##### **Code**
```Python
# Define model
DNN = model_dictionary['Sequential Deep Neural Network']
```

Then, we can add the first two dense layers, both using $\text{ReLU}$ activation functions:

##### **Code**
```Python
# Add first two layers using ReLU activation function
DNN.add(Dense(8, activation = "relu", input_dim = train_Sx.shape[1]))
DNN.add(Dense(16, activation = "relu"))
```

Next, we will add a Dropout regularization layer. A dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time:

##### **Code**
```Python
# Add Dropout regularization layer
DNN.add(Dropout(0.1))
```

We will conclude with our model by adding one last dense $\text{ReLU}$ activation layer, and one dense $\text{softmax}$ layer which will serve as activation for our output layer:

##### **Code**
```Python
# Add third layer using ReLU, and output layer using softmax
DNN.add(Dense(8, activation = "relu"))
DNN.add(Dense(3, activation = "softmax"))
```

We will finally compile our model using $\text{categorical cross entropy}$ as our loss function, and $\text{adam}$ as our optimization function:

##### **Code**
```Python
DNN.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
```

Before training our model, we will need to re-encode & dummify our labels:

##### **Code**
```Python
df_y_D = LabelEncoder().fit_transform(df_y)
df_y_D = pd.get_dummies(df_y_D)
```

We will then fit our model:

##### **Code**
```Python
DNN_Fit = DNN.fit(df_x, df_y_D, epochs = 150, validation_split = 0.3)
```

##### **Output**
```
Epoch 1/150
22/22 [==============================] - 0s 6ms/step - loss: 1.3269 - accuracy: 0.3643 - val_loss: 1.1804 - val_accuracy: 0.4033
Epoch 2/150
22/22 [==============================] - 0s 2ms/step - loss: 1.1279 - accuracy: 0.3957 - val_loss: 1.0665 - val_accuracy: 0.4367
Epoch 3/150
22/22 [==============================] - 0s 2ms/step - loss: 1.0480 - accuracy: 0.4357 - val_loss: 1.0114 - val_accuracy: 0.4700
Epoch 4/150
22/22 [==============================] - 0s 2ms/step - loss: 1.0003 - accuracy: 0.4514 - val_loss: 0.9586 - val_accuracy: 0.4233
Epoch 5/150
22/22 [==============================] - 0s 2ms/step - loss: 0.9571 - accuracy: 0.5271 - val_loss: 0.8948 - val_accuracy: 0.7167
Epoch 6/150
22/22 [==============================] - 0s 2ms/step - loss: 0.8857 - accuracy: 0.6414 - val_loss: 0.8201 - val_accuracy: 0.6367
Epoch 7/150
22/22 [==============================] - 0s 2ms/step - loss: 0.8307 - accuracy: 0.6071 - val_loss: 0.7594 - val_accuracy: 0.6300
Epoch 8/150
22/22 [==============================] - 0s 2ms/step - loss: 0.7781 - accuracy: 0.6443 - val_loss: 0.6931 - val_accuracy: 0.7200
Epoch 9/150
22/22 [==============================] - 0s 2ms/step - loss: 0.7272 - accuracy: 0.6643 - val_loss: 0.6362 - val_accuracy: 0.7367
Epoch 10/150
22/22 [==============================] - 0s 2ms/step - loss: 0.6683 - accuracy: 0.7057 - val_loss: 0.5889 - val_accuracy: 0.7367
Epoch 11/150
22/22 [==============================] - 0s 2ms/step - loss: 0.6249 - accuracy: 0.7471 - val_loss: 0.5391 - val_accuracy: 0.8467
Epoch 12/150
22/22 [==============================] - 0s 2ms/step - loss: 0.5764 - accuracy: 0.7800 - val_loss: 0.4978 - val_accuracy: 0.8267
Epoch 13/150
22/22 [==============================] - 0s 2ms/step - loss: 0.5556 - accuracy: 0.7914 - val_loss: 0.4609 - val_accuracy: 0.8600
Epoch 14/150
22/22 [==============================] - 0s 2ms/step - loss: 0.4973 - accuracy: 0.8271 - val_loss: 0.4248 - val_accuracy: 0.8267
Epoch 15/150
22/22 [==============================] - 0s 2ms/step - loss: 0.4708 - accuracy: 0.8000 - val_loss: 0.3859 - val_accuracy: 0.8267
Epoch 16/150
22/22 [==============================] - 0s 2ms/step - loss: 0.4336 - accuracy: 0.8100 - val_loss: 0.3528 - val_accuracy: 0.8267
Epoch 17/150
22/22 [==============================] - 0s 2ms/step - loss: 0.4099 - accuracy: 0.8271 - val_loss: 0.3288 - val_accuracy: 0.8467
Epoch 18/150
22/22 [==============================] - 0s 2ms/step - loss: 0.3881 - accuracy: 0.8486 - val_loss: 0.3014 - val_accuracy: 0.8767
Epoch 19/150
22/22 [==============================] - 0s 2ms/step - loss: 0.3664 - accuracy: 0.8500 - val_loss: 0.2831 - val_accuracy: 0.8667
Epoch 20/150
22/22 [==============================] - 0s 2ms/step - loss: 0.3305 - accuracy: 0.8586 - val_loss: 0.2621 - val_accuracy: 0.8967
Epoch 21/150
22/22 [==============================] - 0s 2ms/step - loss: 0.3112 - accuracy: 0.8829 - val_loss: 0.2456 - val_accuracy: 0.9000
Epoch 22/150
22/22 [==============================] - 0s 2ms/step - loss: 0.3110 - accuracy: 0.8700 - val_loss: 0.2340 - val_accuracy: 0.9000
Epoch 23/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2728 - accuracy: 0.8871 - val_loss: 0.2196 - val_accuracy: 0.9000
Epoch 24/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2628 - accuracy: 0.9000 - val_loss: 0.2122 - val_accuracy: 0.9000
Epoch 25/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2495 - accuracy: 0.8986 - val_loss: 0.1986 - val_accuracy: 0.9067
Epoch 26/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2574 - accuracy: 0.8914 - val_loss: 0.1865 - val_accuracy: 0.9000
Epoch 27/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2327 - accuracy: 0.8986 - val_loss: 0.1759 - val_accuracy: 0.9067
Epoch 28/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2183 - accuracy: 0.9143 - val_loss: 0.1662 - val_accuracy: 0.9067
Epoch 29/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2102 - accuracy: 0.9086 - val_loss: 0.1666 - val_accuracy: 0.9300
Epoch 30/150
22/22 [==============================] - 0s 2ms/step - loss: 0.2030 - accuracy: 0.9186 - val_loss: 0.1500 - val_accuracy: 0.9533
Epoch 31/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1947 - accuracy: 0.9271 - val_loss: 0.1381 - val_accuracy: 0.9533
Epoch 32/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1783 - accuracy: 0.9314 - val_loss: 0.1294 - val_accuracy: 0.9533
Epoch 33/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1817 - accuracy: 0.9257 - val_loss: 0.1189 - val_accuracy: 0.9833
Epoch 34/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1620 - accuracy: 0.9514 - val_loss: 0.1201 - val_accuracy: 0.9833
Epoch 35/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1648 - accuracy: 0.9500 - val_loss: 0.1146 - val_accuracy: 0.9933
Epoch 36/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1481 - accuracy: 0.9586 - val_loss: 0.1028 - val_accuracy: 0.9933
Epoch 37/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1669 - accuracy: 0.9543 - val_loss: 0.0932 - val_accuracy: 0.9833
Epoch 38/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1263 - accuracy: 0.9771 - val_loss: 0.0851 - val_accuracy: 0.9933
Epoch 39/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1379 - accuracy: 0.9600 - val_loss: 0.0799 - val_accuracy: 0.9933
Epoch 40/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1250 - accuracy: 0.9571 - val_loss: 0.0779 - val_accuracy: 0.9833
Epoch 41/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1099 - accuracy: 0.9786 - val_loss: 0.0719 - val_accuracy: 0.9933
Epoch 42/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1331 - accuracy: 0.9614 - val_loss: 0.0724 - val_accuracy: 0.9933
Epoch 43/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1310 - accuracy: 0.9671 - val_loss: 0.0661 - val_accuracy: 0.9933
Epoch 44/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1210 - accuracy: 0.9671 - val_loss: 0.0630 - val_accuracy: 0.9933
Epoch 45/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0987 - accuracy: 0.9700 - val_loss: 0.0575 - val_accuracy: 1.0000
Epoch 46/150
22/22 [==============================] - 0s 2ms/step - loss: 0.1037 - accuracy: 0.9743 - val_loss: 0.0545 - val_accuracy: 1.0000
Epoch 47/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0807 - accuracy: 0.9857 - val_loss: 0.0491 - val_accuracy: 1.0000
Epoch 48/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0886 - accuracy: 0.9829 - val_loss: 0.0449 - val_accuracy: 1.0000
Epoch 49/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0938 - accuracy: 0.9771 - val_loss: 0.0422 - val_accuracy: 1.0000
Epoch 50/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0836 - accuracy: 0.9871 - val_loss: 0.0376 - val_accuracy: 1.0000
Epoch 51/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0936 - accuracy: 0.9757 - val_loss: 0.0382 - val_accuracy: 1.0000
Epoch 52/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0825 - accuracy: 0.9800 - val_loss: 0.0348 - val_accuracy: 1.0000
Epoch 53/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0798 - accuracy: 0.9814 - val_loss: 0.0380 - val_accuracy: 1.0000
Epoch 54/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0900 - accuracy: 0.9743 - val_loss: 0.0300 - val_accuracy: 1.0000
Epoch 55/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0701 - accuracy: 0.9871 - val_loss: 0.0297 - val_accuracy: 1.0000
Epoch 56/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0673 - accuracy: 0.9814 - val_loss: 0.0290 - val_accuracy: 1.0000
Epoch 57/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0673 - accuracy: 0.9900 - val_loss: 0.0258 - val_accuracy: 1.0000
Epoch 58/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0543 - accuracy: 0.9900 - val_loss: 0.0248 - val_accuracy: 1.0000
Epoch 59/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0619 - accuracy: 0.9857 - val_loss: 0.0218 - val_accuracy: 1.0000
Epoch 60/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0655 - accuracy: 0.9829 - val_loss: 0.0212 - val_accuracy: 1.0000
Epoch 61/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0589 - accuracy: 0.9871 - val_loss: 0.0209 - val_accuracy: 1.0000
Epoch 62/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0598 - accuracy: 0.9871 - val_loss: 0.0192 - val_accuracy: 1.0000
Epoch 63/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0509 - accuracy: 0.9900 - val_loss: 0.0193 - val_accuracy: 1.0000
Epoch 64/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0370 - accuracy: 0.9943 - val_loss: 0.0178 - val_accuracy: 1.0000
Epoch 65/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0395 - accuracy: 0.9929 - val_loss: 0.0159 - val_accuracy: 1.0000
Epoch 66/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0429 - accuracy: 0.9914 - val_loss: 0.0150 - val_accuracy: 1.0000
Epoch 67/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0418 - accuracy: 0.9943 - val_loss: 0.0143 - val_accuracy: 1.0000
Epoch 68/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0369 - accuracy: 0.9957 - val_loss: 0.0160 - val_accuracy: 1.0000
Epoch 69/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0527 - accuracy: 0.9900 - val_loss: 0.0122 - val_accuracy: 1.0000
Epoch 70/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0333 - accuracy: 0.9943 - val_loss: 0.0122 - val_accuracy: 1.0000
Epoch 71/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0517 - accuracy: 0.9914 - val_loss: 0.0125 - val_accuracy: 1.0000
Epoch 72/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0481 - accuracy: 0.9900 - val_loss: 0.0152 - val_accuracy: 1.0000
Epoch 73/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0390 - accuracy: 0.9900 - val_loss: 0.0106 - val_accuracy: 1.0000
Epoch 74/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0280 - accuracy: 0.9957 - val_loss: 0.0103 - val_accuracy: 1.0000
Epoch 75/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0445 - accuracy: 0.9871 - val_loss: 0.0099 - val_accuracy: 1.0000
Epoch 76/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0323 - accuracy: 0.9943 - val_loss: 0.0094 - val_accuracy: 1.0000
Epoch 77/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0442 - accuracy: 0.9871 - val_loss: 0.0093 - val_accuracy: 1.0000
Epoch 78/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0266 - accuracy: 0.9957 - val_loss: 0.0085 - val_accuracy: 1.0000
Epoch 79/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0409 - accuracy: 0.9929 - val_loss: 0.0108 - val_accuracy: 1.0000
Epoch 80/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0423 - accuracy: 0.9857 - val_loss: 0.0100 - val_accuracy: 1.0000
Epoch 81/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0262 - accuracy: 0.9943 - val_loss: 0.0080 - val_accuracy: 1.0000
Epoch 82/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0406 - accuracy: 0.9900 - val_loss: 0.0078 - val_accuracy: 1.0000
Epoch 83/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0330 - accuracy: 0.9929 - val_loss: 0.0079 - val_accuracy: 1.0000
Epoch 84/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0308 - accuracy: 0.9957 - val_loss: 0.0071 - val_accuracy: 1.0000
Epoch 85/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0242 - accuracy: 0.9943 - val_loss: 0.0069 - val_accuracy: 1.0000
Epoch 86/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0335 - accuracy: 0.9900 - val_loss: 0.0066 - val_accuracy: 1.0000
Epoch 87/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0318 - accuracy: 0.9914 - val_loss: 0.0074 - val_accuracy: 1.0000
Epoch 88/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0254 - accuracy: 0.9929 - val_loss: 0.0062 - val_accuracy: 1.0000
Epoch 89/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0304 - accuracy: 0.9943 - val_loss: 0.0067 - val_accuracy: 1.0000
Epoch 90/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0210 - accuracy: 0.9957 - val_loss: 0.0056 - val_accuracy: 1.0000
Epoch 91/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0188 - accuracy: 0.9986 - val_loss: 0.0051 - val_accuracy: 1.0000
Epoch 92/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0213 - accuracy: 0.9957 - val_loss: 0.0051 - val_accuracy: 1.0000
Epoch 93/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0150 - accuracy: 1.0000 - val_loss: 0.0048 - val_accuracy: 1.0000
Epoch 94/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0317 - accuracy: 0.9914 - val_loss: 0.0052 - val_accuracy: 1.0000
Epoch 95/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0370 - accuracy: 0.9957 - val_loss: 0.0050 - val_accuracy: 1.0000
Epoch 96/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0227 - accuracy: 0.9986 - val_loss: 0.0050 - val_accuracy: 1.0000
Epoch 97/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0269 - accuracy: 0.9900 - val_loss: 0.0049 - val_accuracy: 1.0000
Epoch 98/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0316 - accuracy: 0.9886 - val_loss: 0.0045 - val_accuracy: 1.0000
Epoch 99/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0265 - accuracy: 0.9957 - val_loss: 0.0044 - val_accuracy: 1.0000
Epoch 100/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0207 - accuracy: 0.9986 - val_loss: 0.0039 - val_accuracy: 1.0000
Epoch 101/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0192 - accuracy: 0.9971 - val_loss: 0.0038 - val_accuracy: 1.0000
Epoch 102/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0199 - accuracy: 0.9957 - val_loss: 0.0038 - val_accuracy: 1.0000
Epoch 103/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0215 - accuracy: 0.9943 - val_loss: 0.0036 - val_accuracy: 1.0000
Epoch 104/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0161 - accuracy: 0.9986 - val_loss: 0.0036 - val_accuracy: 1.0000
Epoch 105/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0269 - accuracy: 0.9929 - val_loss: 0.0069 - val_accuracy: 1.0000
Epoch 106/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0283 - accuracy: 0.9914 - val_loss: 0.0035 - val_accuracy: 1.0000
Epoch 107/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0169 - accuracy: 0.9986 - val_loss: 0.0038 - val_accuracy: 1.0000
Epoch 108/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0232 - accuracy: 0.9943 - val_loss: 0.0034 - val_accuracy: 1.0000
Epoch 109/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0344 - accuracy: 0.9900 - val_loss: 0.0035 - val_accuracy: 1.0000
Epoch 110/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0167 - accuracy: 0.9971 - val_loss: 0.0035 - val_accuracy: 1.0000
Epoch 111/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0144 - accuracy: 0.9971 - val_loss: 0.0029 - val_accuracy: 1.0000
Epoch 112/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0118 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 1.0000
Epoch 113/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0169 - accuracy: 0.9943 - val_loss: 0.0026 - val_accuracy: 1.0000
Epoch 114/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0165 - accuracy: 0.9943 - val_loss: 0.0026 - val_accuracy: 1.0000
Epoch 115/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0215 - accuracy: 0.9957 - val_loss: 0.0026 - val_accuracy: 1.0000
Epoch 116/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0221 - accuracy: 0.9957 - val_loss: 0.0026 - val_accuracy: 1.0000
Epoch 117/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0190 - accuracy: 0.9886 - val_loss: 0.0030 - val_accuracy: 1.0000
Epoch 118/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0158 - accuracy: 0.9971 - val_loss: 0.0025 - val_accuracy: 1.0000
Epoch 119/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0147 - accuracy: 0.9957 - val_loss: 0.0026 - val_accuracy: 1.0000
Epoch 120/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0121 - accuracy: 0.9986 - val_loss: 0.0038 - val_accuracy: 1.0000
Epoch 121/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0157 - accuracy: 0.9971 - val_loss: 0.0034 - val_accuracy: 1.0000
Epoch 122/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0164 - accuracy: 0.9971 - val_loss: 0.0022 - val_accuracy: 1.0000
Epoch 123/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0140 - accuracy: 0.9957 - val_loss: 0.0022 - val_accuracy: 1.0000
Epoch 124/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0203 - accuracy: 0.9914 - val_loss: 0.0021 - val_accuracy: 1.0000
Epoch 125/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0118 - accuracy: 1.0000 - val_loss: 0.0020 - val_accuracy: 1.0000
Epoch 126/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0131 - accuracy: 0.9971 - val_loss: 0.0035 - val_accuracy: 1.0000
Epoch 127/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0100 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 1.0000
Epoch 128/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0199 - accuracy: 0.9943 - val_loss: 0.0020 - val_accuracy: 1.0000
Epoch 129/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0162 - accuracy: 0.9971 - val_loss: 0.0020 - val_accuracy: 1.0000
Epoch 130/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0109 - accuracy: 0.9971 - val_loss: 0.0019 - val_accuracy: 1.0000
Epoch 131/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0100 - accuracy: 0.9986 - val_loss: 0.0017 - val_accuracy: 1.0000
Epoch 132/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0130 - accuracy: 0.9943 - val_loss: 0.0016 - val_accuracy: 1.0000
Epoch 133/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0146 - accuracy: 0.9971 - val_loss: 0.0018 - val_accuracy: 1.0000
Epoch 134/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0193 - accuracy: 0.9943 - val_loss: 0.0014 - val_accuracy: 1.0000
Epoch 135/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0111 - accuracy: 0.9986 - val_loss: 0.0016 - val_accuracy: 1.0000
Epoch 136/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0190 - accuracy: 0.9943 - val_loss: 0.0021 - val_accuracy: 1.0000
Epoch 137/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0103 - accuracy: 0.9971 - val_loss: 0.0016 - val_accuracy: 1.0000
Epoch 138/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0174 - accuracy: 0.9929 - val_loss: 0.0014 - val_accuracy: 1.0000
Epoch 139/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0122 - accuracy: 0.9971 - val_loss: 0.0014 - val_accuracy: 1.0000
Epoch 140/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0123 - accuracy: 0.9957 - val_loss: 0.0013 - val_accuracy: 1.0000
Epoch 141/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0138 - accuracy: 0.9943 - val_loss: 0.0014 - val_accuracy: 1.0000
Epoch 142/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0121 - accuracy: 0.9986 - val_loss: 0.0013 - val_accuracy: 1.0000
Epoch 143/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0102 - accuracy: 0.9986 - val_loss: 0.0011 - val_accuracy: 1.0000
Epoch 144/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0081 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
Epoch 145/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0116 - accuracy: 0.9971 - val_loss: 0.0012 - val_accuracy: 1.0000
Epoch 146/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0105 - accuracy: 0.9971 - val_loss: 0.0011 - val_accuracy: 1.0000
Epoch 147/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0074 - accuracy: 0.9986 - val_loss: 0.0012 - val_accuracy: 1.0000
Epoch 148/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0122 - accuracy: 0.9957 - val_loss: 9.7937e-04 - val_accuracy: 1.0000
Epoch 149/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0118 - accuracy: 0.9986 - val_loss: 0.0010 - val_accuracy: 1.0000
Epoch 150/150
22/22 [==============================] - 0s 2ms/step - loss: 0.0081 - accuracy: 0.9986 - val_loss: 9.0221e-04 - val_accuracy: 1.0000
```

This output shows us each epoch, the training accuracy achieved, the validation accuracy achieved, and the loss for each step. Here we can see that 150 epochs was barely enough for our model to achieve a training accuracy of 100%. We can also see that in the beginning, our accuracy increases in a higher rate; this makes sense since the gradient for the first epochs is bigger, as we'll confirm shortly using visualization methods.

We can get a model summary:

##### **Code**
```Python
# Get model summary
DNN.summary()
```

##### **Output**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 8)                 88                                                                  
 dense_1 (Dense)             (None, 16)                144                                                                
 dropout (Dropout)           (None, 16)                0                                                                    
 dense_2 (Dense)             (None, 8)                 136                                                                  
 dense_3 (Dense)             (None, 3)                 27                                                                   
=================================================================
Total params: 395
Trainable params: 395
Non-trainable params: 0
_________________________________________________________________
```

We can visualize the 5 layers we specified previously, their shape, the number of parameters (*inputs*) for each layer and the total number of trainable parameters.

We can also create two plots that will help us visualize how our model performed. The first one will plot epochs on the $x$ axis and training accuracy & validation accuracy on the $y$ axis. The second one will plot epochs on the $x$ axis and training & validation loss on the $y$ axis:

##### **Code**
```Python
# Plot epochs vs training accuracy & validation accuracy
plt.figure('Epochs vs Accuracy')
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy of Data")
plt.plot(DNN_Fit.history["accuracy"], label = "Training Accuracy", color = 'k', linewidth = 0.7, marker = 'o', markersize=2)
plt.plot(DNN_Fit.history["val_accuracy"],label = "Validation Accuracy", color = '#24c98d', linewidth = 0.7, marker = 'o', markersize=2)
plt.title("Training Vs. Validation Accuracy")
plt.legend()
plt.savefig('plots/' + 'G001A008_Deep Neural Network_Epochs vs Accuracy.png', format = 'png', dpi = 300, transparent = True)
plt.close()

# Plot training vs validation loss
plt.figure('Training vs Validation Loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss in Data")
plt.plot(DNN_Fit.history["loss"], label= "Training Loss", color = 'k', linewidth = 0.7, marker = 'o', markersize=2)
plt.plot(DNN_Fit.history["val_loss"], label= "Validation Loss", color = '#24c98d', linewidth = 0.7, marker = 'o', markersize=2)
plt.title("Training Vs. Validation loss")
plt.legend()
plt.savefig('plots/' + 'G001A008_Deep Neural Network_Training vs Validation Loss.png', format = 'png', dpi = 300, transparent = True)
plt.close()
```

##### **Output**

IMAGE
IMAGE


### 17. Method comparison

## Conclusions


## References
- [IBM, KNN](https://www.ibm.com/topics/knn)
- [OpenGenus, Gaussian Naïve Bayes](https://iq.opengenus.org/gaussian-naive-bayes/)
- [OpenGenus, Bernoulli Naïve Bayes](https://iq.opengenus.org/bernoulli-naive-bayes/)
- [Michael Fuchs, Introduction to SGD Classifier](https://michael-fuchs-python.netlify.app/2019/11/11/introduction-to-sgd-classifier/)
- [Jerry Friedman, Greedy Function Approximation: A Gradient Boosting Machine](https://jerryfriedman.su.domains/ftp/trebst.pdf)
- [Machine Learning Mastery, Multinomial Logistic Regression With Python](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/)
- [Philipp Christian Petersen, Neural Network Theory](http://pc-petersen.eu/Neural_Network_Theory.pdf)


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
- https://www.educative.io/answers/implement-neural-network-for-classification-using-scikit-learn
