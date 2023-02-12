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
import xlsxwriter
import re

# Plotting modules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import tree
import visualkeras
from PIL import ImageFont

# Preprocessing modules
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, KFold

# Evaluation & performance modules
from sklearn.metrics import confusion_matrix, classification_report

# Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
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
    'Multinomial Logistic Regressor': LogisticRegression(multi_class='multinomial',
                                                         solver='lbfgs',
                                                         random_state=42,
                                                         max_iter=100000,
                                                         penalty='l2',
                                                         C=24),
    'Logistic Regressor' : LogisticRegression(C=24),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=9),
    'Random Forest Classifier': RandomForestClassifier(n_estimators = 100),
    'Support Vector Classifier': SVC(C=0.12, gamma=0.02, kernel='linear'),
    'Support Vector Classifier Polynomial Kernel': SVC(C=1, gamma=0.6, kernel='poly', degree=8),
    'Support Vector Classifier Radial Kernel': SVC(C=1, gamma=0.6, kernel='rbf'),
	'K-Nearest Neighbors Classifier' : KNeighborsClassifier(n_neighbors=5),
    'Gaussian Naive Bayes Classifier': GaussianNB(),
    'Bernoulli Naive Bayes Classifier': BernoulliNB(),
    'Stochastic Gradient Descent': SGDClassifier(loss='log',
                                                 max_iter=10000,
                                                 random_state=42,
                                                 penalty='l2'),
	'Gradient Boosting Classifier': GradientBoostingClassifier(),
	'Extreme Gradient Boosting Classifier' : XGBClassifier(),
    'Sequential Deep Neural Network' : Sequential()
}

# Define preprocessing functions
preprocessing_dictionary = {'Right Skew Gaussian' : FunctionTransformer(func = np.square),
                            'Left Skew Gaussian' : FunctionTransformer(func = np.log1p),
                            'Standard Scaler' : StandardScaler()
}

# Preprocessing
# --------------------

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

# Separate labels from features by defining a function
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
    plt.savefig('plots/' + f'G001A008_{model_name}_confusion_matrix_tp.png', format = 'png', dpi = 300, transparent = True)
    plt.savefig('plots/' + f'G001A008_{model_name}_confusion_matrix_bg.jpg', format = 'jpg', dpi = 300, transparent = False)
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

# Transforming Data
# --------------------

df_x, df_y = sep(df)

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

# Test for balanced label class
df.groupby('Level')['Level'].count() / len(df) * 100


# Multinomial Logistic Regression
# --------------------
# Train model
model_dictionary['Multinomial Logistic Regressor'].fit(train_Sx, train_Sy)

# Predict
y_predicted_MLogReg = model_dictionary['Multinomial Logistic Regressor'].predict(test_Sx)

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

# Binomial Logistic Regression
# --------------------
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


# Decision Tree Classifier
# --------------------
# Train model
model_dictionary['Decision Tree Classifier'].fit(train_Sx, train_Sy)

# Predict
y_predicted_DecTree = model_dictionary['Decision Tree Classifier'].predict(test_Sx)

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

# Visualize the Decision Tree

# Text Representation
DecTree_text_rep = tree.export_text(model_dictionary['Decision Tree Classifier'])

# Tree plot using plot_tree
fig = plt.figure('Decision Tree plot_tree')
tree.plot_tree(model_dictionary['Decision Tree Classifier'],
                   feature_names=df_x.columns,
                   class_names=df_y['Level'].astype('str'),
                   filled=True)

plt.title('Decision Tree Plot')
plt.savefig('plots/' + 'G001A008_Decision Tree Classifier_Decision Tree_tp.png', format = 'png', dpi = 300, transparent = True)
plt.savefig('plots/' + 'G001A008_Decision Tree Classifier_Decision Tree_bg.jpg', format = 'jpg', dpi = 300, transparent = False)
plt.close()


# Random Forest Classifier
# --------------------
# Train model
model_dictionary['Random Forest Classifier'].fit(train_Sx, train_Sy)

# Predict
y_predicted_RandomFor = model_dictionary['Random Forest Classifier'].predict(test_Sx)

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

# Linear Support Vector Machine Classifier
# --------------------
# Train model
model_dictionary['Support Vector Classifier'].fit(train_Sx, train_Sy)

# Predict
y_predicted_SVM = model_dictionary['Support Vector Classifier'].predict(test_Sx)

# Evaluate the model and collect the scores
cm_plot('Support Vector Classifier',
        model_dictionary['Support Vector Classifier'],
        test_Sy,
        y_predicted_SVM)

# Define model score
score_SVM = model_score(model_dictionary['Support Vector Classifier'],
                        test_Sx,
                        test_Sy)

# Define Classification Report Function
report_SVM = classification_rep(test_Sy,
                                y_predicted_SVM)

print(score_SVM)


# Support Vector Machine Classifier - Polynomial Kernel
# --------------------
# Train model
model_dictionary['Support Vector Classifier Polynomial Kernel'].fit(train_Sx, train_Sy)

# Predict
y_predicted_SVMp = model_dictionary['Support Vector Classifier Polynomial Kernel'].predict(test_Sx)

# Evaluate the model and collect the scores
cm_plot('Support Vector Classifier Polynomial Kernel',
        model_dictionary['Support Vector Classifier Polynomial Kernel'],
        test_Sy,
        y_predicted_SVMp)

# Define model score
score_SVMp = model_score(model_dictionary['Support Vector Classifier Polynomial Kernel'],
                        test_Sx,
                        test_Sy)

# Define Classification Report Function
report_SVMp = classification_rep(test_Sy,
                                y_predicted_SVMp)

print(score_SVMp)


# Support Vector Machine Classifier - Radial Basis Function Kernel
# --------------------
# Train model
model_dictionary['Support Vector Classifier Radial Kernel'].fit(train_Sx, train_Sy)

# Predict
y_predicted_SVMr = model_dictionary['Support Vector Classifier Radial Kernel'].predict(test_Sx)

# Evaluate the model and collect the scores
cm_plot('Support Vector Classifier Radial Kernel',
        model_dictionary['Support Vector Classifier Radial Kernel'],
        test_Sy,
        y_predicted_SVMr)

# Define model score
score_SVMr = model_score(model_dictionary['Support Vector Classifier Radial Kernel'],
                        test_Sx,
                        test_Sy)

# Define Classification Report Function
report_SVMr = classification_rep(test_Sy,
                                y_predicted_SVMr)

print(score_SVMr)


# K-Nearest Neighbours
# --------------------
# Train model
model_dictionary['K-Nearest Neighbors Classifier'].fit(train_Sx, train_Sy)

# Predict
y_predicted_KNN = model_dictionary['K-Nearest Neighbors Classifier'].predict(test_Sx)

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


# Gaussian Naive Bayes Classifier
# --------------------
# Train model
model_dictionary['Gaussian Naive Bayes Classifier'].fit(train_Gx, train_Gy)

# Predict
y_predicted_GNB = model_dictionary['Gaussian Naive Bayes Classifier'].predict(test_Gx)

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


# Bernoulli Naive Bayes Classifier
# --------------------
# Train model
model_dictionary['Bernoulli Naive Bayes Classifier'].fit(train_Sx, train_Sy)

# Predict
y_predicted_BNB = model_dictionary['Bernoulli Naive Bayes Classifier'].predict(test_Sx)

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


# Stochastic Gradient Descent
# --------------------
# Train model
model_dictionary['Stochastic Gradient Descent'].fit(train_Sx, train_Sy)

# Predict
y_predicted_SGD = model_dictionary['Stochastic Gradient Descent'].predict(test_Sx)

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


# Gradient Boosting Classifier
# --------------------
# Train model
model_dictionary['Gradient Boosting Classifier'].fit(train_Sx, train_Sy)

# Predict
y_predicted_GBC = model_dictionary['Gradient Boosting Classifier'].predict(test_Sx)

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


# Extreme Gradient Boosting Classifier
# --------------------
# Re-encode labels
train_Sy_XGBC = LabelEncoder().fit_transform(train_Sy)
test_Sy_XGBC = LabelEncoder().fit_transform(test_Sy)

# Train model
model_dictionary['Extreme Gradient Boosting Classifier'].fit(train_Sx, train_Sy_XGBC)

# Predict
y_predicted_XGBC = model_dictionary['Extreme Gradient Boosting Classifier'].predict(test_Sx)

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


# Deep Neural Network
# --------------------

# Define model
DNN = model_dictionary['Sequential Deep Neural Network']

# Add first two layers using ReLU activation function
DNN.add(Dense(8, activation = "relu", input_dim = train_Sx.shape[1]))
DNN.add(Dense(16, activation = "relu"))

# Add Dropout regularization layer
DNN.add(Dropout(0.1))

# Add third layer using ReLU, and output layer using softmax
DNN.add(Dense(8, activation = "relu"))
DNN.add(Dense(3, activation = "softmax"))

# Compile our model
DNN.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Re-encode & dummify labels
df_y_D = LabelEncoder().fit_transform(df_y)
df_y_D = pd.get_dummies(df_y_D)

# Fit our compiled model
DNN_Fit = DNN.fit(df_x, df_y_D, epochs = 150, validation_split = 0.3)

# Print model summary
DNN.summary()

# Convert model summary to DataFrame object
stringlist = []
DNN.summary(print_fn=lambda x: stringlist.append(x))
summ_string = "\n".join(stringlist)
print(summ_string)

table = stringlist[1:-4][1::2]

new_table = []
for entry in table:
    entry = re.split(r'\s{2,}', entry)[:-1]
    new_table.append(entry)

DNN_summary = pd.DataFrame(new_table[1:], columns=new_table[0])

# Plot epochs vs training accuracy & validation accuracy
plt.figure('Epochs vs Accuracy')
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy of Data")
plt.plot(DNN_Fit.history["accuracy"], label = "Training Accuracy", color = 'k', linewidth = 0.7, marker = 'o', markersize=2)
plt.plot(DNN_Fit.history["val_accuracy"],label = "Validation Accuracy", color = '#24c98d', linewidth = 0.7, marker = 'o', markersize=2)
plt.title("Training Vs. Validation Accuracy")
plt.legend()
plt.savefig('plots/' + 'G001A008_Deep Neural Network_Epochs vs Accuracy_tp.png', format = 'png', dpi = 300, transparent = True)
plt.savefig('plots/' + 'G001A008_Deep Neural Network_Epochs vs Accuracy_bg.jpg', format = 'jpg', dpi = 300, transparent = False)
plt.close()

# Plot training vs validation loss
plt.figure('Training vs Validation Loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss in Data")
plt.plot(DNN_Fit.history["loss"], label= "Training Loss", color = 'k', linewidth = 0.7, marker = 'o', markersize=2)
plt.plot(DNN_Fit.history["val_loss"], label= "Validation Loss", color = '#24c98d', linewidth = 0.7, marker = 'o', markersize=2)
plt.title("Training Vs. Validation loss")
plt.legend()
plt.savefig('plots/' + 'G001A008_Deep Neural Network_Training vs Validation Loss_tp.png', format = 'png', dpi = 300, transparent = True)
plt.savefig('plots/' + 'G001A008_Deep Neural Network_Training vs Validation Loss_bg.jpg', format = 'jpg', dpi = 300, transparent = False)
plt.close()

# Plot Deep Neural Network structure

# Using visualkeras layered_view
visualkeras.layered_view(DNN,
                         legend=True,
                         to_file='plots/' + 'G001A008_Deep Neural Network_Model.png',
                         scale_z=5,
                         spacing=30)

# K-Fold Cross Validation

# Define 10-fold cross validation test harness
kfold = KFold(n_splits=10, shuffle=True)
DNN_accuracy_scores = []
DNN_loss = []
for train, test in kfold.split(df_x, df_y_D):
    # Create model
    DNN = Sequential()
    DNN.add(Dense(8, activation = "relu", input_dim = train_Sx.shape[1]))
    DNN.add(Dense(16, activation = "relu"))
    DNN.add(Dropout(0.1))
    DNN.add(Dense(8, activation = "relu"))
    DNN.add(Dense(3, activation = "softmax"))
    DNN.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    # Fit the model
    DNN.fit(df_x.iloc[train], df_y_D.iloc[train], epochs=150, verbose=0)
    # Evaluate the model
    scores = DNN.evaluate(df_x.iloc[test], df_y_D.iloc[test], verbose=0)
    print(f'{DNN.metrics_names[0]}: {round(scores[0]*100, 2)}%')
    print(f'{DNN.metrics_names[1]}: {round(scores[1]*100, 2)}%')
    print('')
    DNN_loss.append(scores[0])
    DNN_accuracy_scores.append(scores[1])

print(f'{round(np.mean(DNN_accuracy_scores),2)*100}%, +/-{round(np.std(DNN_accuracy_scores),2)*100}%')

DNN_scores_df = pd.DataFrame(columns=['Accuracy', 'Loss'])
DNN_scores_df['Accuracy'] = DNN_accuracy_scores
DNN_scores_df['Loss'] = DNN_loss

# Export results to Excel sheet
# Create a DataFrame containing accuracy values
acc_list = [score_MLogReg,
            score_BLogReg,
            score_DecTree,
            score_RandomFor,
            score_SVM,
            score_SVMp,
            score_SVMr,
            score_KNN,
            score_GNB,
            score_BNB,
            score_SGD,
            score_GBC,
            score_XGBC,
            round(np.mean(DNN_accuracy_scores),2)
            ]

model_list = list(model_dictionary.keys())

acc_df = pd.DataFrame(columns=['Model', 'Accuracy'])
acc_df['Model'] = model_list
acc_df['Accuracy'] = acc_list

# Write to Excel document
writer = pd.ExcelWriter('outputs/Model_Results.xlsx', engine = 'xlsxwriter')
report_MLogReg.to_excel(writer, sheet_name = 'REP_MLOGREG')
report_BLogReg.to_excel(writer, sheet_name = 'REP_BLOGREG')
report_DecTree.to_excel(writer, sheet_name = 'REP_DECTREE')
report_RandomFor.to_excel(writer, sheet_name = 'REP_RANDOMFOR')
report_SVM.to_excel(writer, sheet_name = 'REP_SVMLIN')
report_SVMp.to_excel(writer, sheet_name = 'REP_SVMPOL')
report_SVMr.to_excel(writer, sheet_name = 'REP_SVMRAD')
report_KNN.to_excel(writer, sheet_name = 'REP_KNN')
report_GNB.to_excel(writer, sheet_name = 'REP_GNB')
report_BNB.to_excel(writer, sheet_name = 'REP_BNB')
report_SGD.to_excel(writer, sheet_name = 'REP_SGD')
report_GBC.to_excel(writer, sheet_name = 'REP_GBC')
report_XGBC.to_excel(writer, sheet_name = 'REP_XGBC')
DNN_summary.to_excel(writer, sheet_name = 'SUM_DNN')
DNN_scores_df.to_excel(writer, sheet_name = 'REP_DNN')
acc_df.to_excel(writer, sheet_name = 'ACC_ALL')
writer.close()