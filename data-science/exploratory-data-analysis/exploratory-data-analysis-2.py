# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:15:12 2023
@author: Pablo Aguirre
GitHub: https://github.com/pabloagn
Website: https://pabloagn.com
Contact: https://pabloagn.com/contact
Part of Guided Project: exploratory-data-analysis
"""

"""
Problem Statement
-----------------
Our client, an insurance company, has asked us to conduct a correlational study
between different behavioral & genetic characteristics and lung cancer incidence.
For this, they have provided a medical dataset containing a set of anonymous patients
along with their medical file generated upon hospital admission.
"""

"""
Data Set
-----------------
https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
"""

# Preparing the data
# ---------------------------

# Import required modules
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

# Define model dictionary
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

