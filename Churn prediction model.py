#!/usr/bin/env python
# coding: utf-8

# In[22]:


#importing the basic packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split ,PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report ,confusion_matrix, accuracy_score


# In[4]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn')


# In[2]:


#loading the dataset

df = pd.read_csv("C:\\Users\\bhara\\Downloads\\Churn_Modelling.csv")
df.head()


# # EDA and Preprocessing

# In[3]:


#Checking the shape of the dataset
df.shape


# In[4]:


#Listing the features of the dataset
df.columns


# In[14]:


df=df.drop('RowNumber',axis=1)


# In[15]:


#Information about the dataset
df.info()


# In[16]:


#Checking for any null values in dataset
df.isnull().sum()


# In[17]:


#checking for any duplicated values in the dataset
df[df.duplicated()]


# In[18]:


#Plotting the data distribution
df.hist(bins = 50,figsize = (20,20))
plt.show()


# In[127]:


#Correlation Heatmap

# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])

# Set up the matplotlib figure
plt.figure(figsize=(15, 13))

# Draw the heatmap with the numeric data
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .8})

# Add title to the heatmap
plt.title('Correlation Heatmap', fontsize=18)

# Show the plot
plt.show()


# In[24]:


sns.pairplot(df)
plt.show()


# In[8]:


df.describe()


# In[29]:


#label Encoder

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'],drop_first= True)


# In[30]:


df.head()


# In[31]:


features = ['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard',
            'IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain']
X= df[features]
y= df['Exited']


# # Splitting the dataset into training and testing dataset

# In[32]:


# Splitting the dataset into train and test sets: 80-20 split

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.2 , random_state=42)
X_train.shape, X_test.shape


# In[33]:


#Scaling the dataset

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[34]:


X_train[:5],X_test[:5]


# This dataset is used for a supervised machine learning task, specifically a classification problem. The goal is to predict customer churn, where the target variable indicates whether a customer has churned (1) or not (0). The classification models considered for training on this dataset include:
# 
#   1) Random Forest
#   2) Logistic Regression
#   3) Support Vector Machine (SVM)
#   4) K-Nearest Neighbor (KNN)
#   5) Gradient Boosting Classfier

# # Random Forest

# We are using the Random Forest model in this churn prediction task because it is a robust ensemble learning method
# that combines multiple decision trees to improve predictive accuracy and reduce overfitting. It can handle large 
# datasets with high dimensionality and automatically manages interactions between features, making it well-suited 
# for complex classification problems like churn prediction. Additionally, Random Forest provides feature importance
# scores, which help in understanding the impact of different variables on the model's predictions.

# In[35]:


#Random Forest Classfier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)


# In[36]:


y_pred= model.predict(X_test)


# In[37]:


#computing the accuracy of the model performance

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)


# In[38]:


#printing the acuuracy results 

print(conf_matrix)
print(class_report)
print(accuracy)


# In[90]:


#Plotting the important features 

importances = model.feature_importances_  
indices = np.argsort(importances)[::-1]
names = [features[i] for i in indices]

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]), importances[indices])
plt.yticks(range(X.shape[1]), names)  # 'x' should be 'X' to match your dataset
plt.gca().invert_yaxis()  # Optional: Invert y-axis for better visualization
plt.show()


# # Logistic Regression

# We are using Logistic Regression in this churn prediction task because it’s a straightforward model that offers interpretable results, making it easy to understand the factors influencing churn. It’s effective for binary classification and provides
# quick training and deployment, especially when there’s a clear linear relationship between features and the target variable.

# In[39]:


#Logistic regression model

from sklearn.linear_model import LogisticRegression

#Build and train the logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

#Make Predictions
y_pred_log_reg = log_reg.predict(X_test)

#Evaluate the Model
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

#printing the results
print(conf_matrix_log_reg , class_report_log_reg ,accuracy_log_reg)


# # Support Vector Machine

# We are using Support Vector Machine (SVM) in this churn prediction task because it is effective at finding the optimal boundary
# between classes, even in high-dimensional spaces. SVM is particularly useful for handling complex data with non-linear 
# relationships, thanks to its ability to apply kernel functions. It also aims to maximize the margin between classes, 
# improving the model's generalization to unseen data.

# In[40]:


#Support Vector Machine model
from sklearn.svm import SVC

#Build and train the SVM model
svm_model= SVC(kernel= 'linear',random_state=42)
svm_model.fit(X_train , y_train)

#Make Predictions
y_pred_svm = svm_model.predict(X_test)

#Evaluate the model
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm =classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test , y_pred_svm)

#printing the results
print(conf_matrix_svm,class_report_svm, accuracy_svm)


# # K Nearest Neighbor (KNN) 

# We are using K-Nearest Neighbors (KNN) in this churn prediction task because it’s a simple, non-parametric model that makes predictions based on the closest data points in the feature space. KNN is effective for capturing local patterns and relationships in the data, making it useful for datasets where similar customers are likely to exhibit similar behavior. Additionally, KNN is easy to understand and implement, with no assumptions about the underlying data distribution.

# In[95]:


from sklearn.neighbors import KNeighborsClassifier

#Build and train the KNN model
knn_model= KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)

#Make Predictions
y_pred_knn = knn_model.predict(X_test)

#Evaluate the model 
conf_matrix_knn = confusion_matrix(y_test , y_pred_knn)
class_report_knn= classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test , y_pred_knn)

#printing the results
print(conf_matrix_knn, class_report_knn ,accuracy_knn)


# # Gradient Boosting Classifier

# We are using Gradient Boosting Classifier in this churn prediction task because it is a powerful ensemble technique that builds models sequentially, each correcting the errors of the previous one. This approach allows it to achieve high accuracy by combining multiple weak learners into a strong predictor. Gradient Boosting is effective at handling complex patterns in data and is particularly strong in minimizing bias and variance, making it well-suited for predicting customer churn.

# In[103]:


#Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

#Build and train the Gradient Boosting Model
gbm_model = GradientBoostingClassifier(n_estimators=100,random_state=42)
gbm_model.fit(X_train , y_train)

#Make predictions 
y_pred_gbm = gbm_model.predict(X_test)

#Evaluate the model
conf_matrix_gbm = confusion_matrix(y_test, y_pred_gbm)
class_report_gbm = classification_report(y_test , y_pred_gbm)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)

#printing the results 
print(conf_matrix_gbm , class_report_gbm, accuracy_gbm)


# # MODEL COMPARISON

# # Feature Selection

# We have performed feature selection in our model to enhance its performance by identifying and retaining the most relevant features for predicting churn. This process helps in reducing dimensionality, improving model accuracy, and decreasing computation time by eliminating redundant or irrelevant features. Effective feature selection ensures that the model focuses on the most impactful variables, leading to more robust and interpretable results.

# In[104]:


#feature enginnering

import pandas as pd

# Load the dataset
df = pd.read_csv("C:\\Users\\bhara\\Downloads\\Churn_Modelling.csv")

# Binary feature for balance (1 if balance is zero, else 0)
df['BalanceZero'] = (df['Balance'] == 0).astype(int)

# Age Groups
df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[18, 25, 35, 45, 55, 65, 75, 85, 95], 
                        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95'])

# Balance to Salary Ratio
df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']

# Interaction feature between NumOfProducts and IsActiveMember
df['ProductUsage'] = df['NumOfProducts'] * df['IsActiveMember']

# Tenure grouping
df['TenureGroup'] = pd.cut(df['Tenure'], 
                           bins=[0, 2, 5, 7, 10], 
                           labels=['0-2', '3-5', '6-7', '8-10'])


# In[105]:


#label Encoding 

label_encoder=LabelEncoder()
df['Gender']=label_encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df,columns=['Geography'], drop_first=True)
df['Male_Germany']=df['Gender']*df['Geography_Germany']
df['Male_Spain'] = df['Gender']*df['Geography_Spain']


# In[107]:


#one hot encoding for 'AgeGroup' and 'TenureGroup'
df = pd.get_dummies(df, columns=['AgeGroup','TenureGroup'],drop_first=True)


# In[109]:


#identifying the important features
features = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary',
            'Geography_Germany','Geography_Spain','BalanceZero','BalanceToSalaryRatio','ProductUsage','Male_Germany',
            'Male_Spain'] +[col for col in df.columns if 'AgeGroup_' in col or 'TenureGroup_' in col]
X = df[features]
y = df['Exited']


# In[113]:


# Splitting the dataset into train and test sets: 80-20 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[115]:


#scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Random Forest Classifier after feature selection
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[116]:


#computing the accuracy of the model 
performanceconf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)


# In[117]:


#printing the results 
print(conf_matrix)
print(class_report)
print(accuracy)


# In[41]:


#Logistic regression model


#Build and train the logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

#Make Predictions
y_pred_log_reg = log_reg.predict(X_test)

#Evaluate the Model
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

#printing the results
print(conf_matrix_log_reg , class_report_log_reg ,accuracy_log_reg)

