#!/usr/bin/env python
# coding: utf-8

# In[1]:


#---------------------------------------------------------------
# @author Prasanth Moothedath Padmakumar C0796752
# @author Aswathy Kuttisseril Jewel C0813455
# @author Gayathri Ravi Nath C0818959

# Dataset File downloaded from https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

# Dataset is used predict weather a person has heart deasease or not based on 13 predictors


# ### Import Libraries

# In[1]:


# Pandas library for data manipulation and analysis
# Numpy library for some standard mathematical functions
# Matplotlib library to visualize the data in the form of different plot
# Seaborn library for visualizing statistical graphics and work on top of Matplotlib
# sklearn for random forest classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
# To display plot within the document
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read the dataset from csv file

# In[2]:


df = pd.read_csv('heart_cleveland_upload.csv')


# In[4]:


# Display first 5 rows of the dataset using head function
df.head()


# ### Checking data set for null values and duplicate entries

# In[5]:


# Print summary of the dataframe 
df.info( )


# In[6]:


# Checking count of duplicate entries in the data set
df.duplicated().sum()


# In[7]:


# Checking missing values in data set
df.isna().sum()


# In[8]:


# Checking null values in data set
df.isnull().sum()


# In[9]:


# Describing the data set 
df.describe( )


# In[10]:


# Display columns in the data set
df.columns


# In[11]:


# Rename columns to readable names
df.columns = ['Age', 'Sex', 'Chest_Pain_Type', 'Resting_Blood_Pressure', 'Cholesterol', 'Fasting_Blood_Sugar', 'Rest_Ecg', 'Max_Heart_Rate_Achieved',
             'Exercise_Induced_Angina', 'St_Depression', 'St_Slope', 'Num_major_vessels', 'Thalassemia', 'Heart_Disease']
df.columns


# ### Exploratory analysis

# #### Age comparison

# In[12]:


sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
sns.catplot(kind = 'count', data = df, x = 'Age', hue = 'Heart_Disease', order = df['Age'].sort_values().unique(), height=5, aspect=5)
plt.title('Variation of Age for each target class')
plt.show()


# The age which people suffer heart disease the most is 58 followed by 57

# In[13]:


minAge=min(df.Age)
maxAge=max(df.Age)
meanAge=df.Age.mean()
print('Min Age :',minAge)
print('Max Age :',maxAge)
print('Mean Age :',meanAge)


# In[14]:


Young = df[(df.Age>=29)&(df.Age<40)&(df.Heart_Disease==1)]
Middle = df[(df.Age>=40)&(df.Age<55)&(df.Heart_Disease==1)]
Elder = df[(df.Age>55)&(df.Heart_Disease==1)]

# Plot as bargraph to find out which age group is most affected to heart disease
plt.figure(figsize=(23,10))
sns.set_context('notebook',font_scale = 1.5)
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(Young),len(Middle),len(Elder)])
plt.tight_layout()


# In[15]:


# Plot it as pie chart
colors = ['blue','green','yellow']
explode = [0,0,0.1]
plt.figure(figsize=(10,10))
sns.set_context('notebook',font_scale = 1.2)
plt.pie([len(Young),len(Middle),len(Elder)],labels=['Young ages','Middle ages','Elderly ages'],explode=explode,colors=colors, autopct='%1.1f%%')
plt.tight_layout()


# #### Age and gender distribution for each target class

# In[16]:


sns.catplot(kind = 'bar', data = df, y = 'Age', x = 'Sex', hue = 'Heart_Disease')
plt.title('Distribution of age vs sex with the target class')
plt.show()


# We see that for females who are suffering from the disease are older than males.

# #### Comparing Chest pain types

# In[17]:


sns.countplot(data= df, x='Chest_Pain_Type',hue='Heart_Disease')
plt.title('Chest Pain Type v/s Heart_Disease\n')


# asymptomatic chest pain is the most common in heart disease patients

# #### Relating heart disease with Thalassemia

# In[18]:


sns.countplot(data= df, x='Sex',hue='Thalassemia')
plt.title('Gender v/s Thalassemia\n')
print('Threed different types of Thalassemia : normal (0), fixed defect (1) ,reversable defect (2)')


# Most females are showing Normal Thalassemia, while majority of males show reversable defect Thalassemia

# In[19]:


pd.crosstab(df['Thalassemia'],df['Heart_Disease']).plot(kind='bar')
plt.title("Heart Disease Frequency per Thalassemia_Types")
plt.xlabel("Thalassemia Types")
plt.ylabel("Amount")
plt.legend(['No disease','Disease'])
plt.xticks(rotation=0);


# Here we can see type 2 Thalassemia patients have most chance of heart Disease

# #### Comparing ECG result and heart disease

# In[20]:


pd.crosstab(df['Rest_Ecg'],df['Heart_Disease']).plot(kind='bar')
plt.title("Heart Disease Frequency per ECG Results")
plt.xlabel("ECG Result types")
plt.ylabel("Amount")
plt.legend(['No disease','Disease'])
plt.xticks(rotation=0);


# Type 1 ECG result type is more prone to heart disease

# #### Comparing fasting blood sugar and heart disease

# In[21]:


pd.crosstab(df['Heart_Disease'],df['Fasting_Blood_Sugar']).plot(kind="bar",figsize=(10,6));
plt.title("Heart Disease Frequency vs Fasting Blood Sugar")
plt.xlabel("0 = No Disease , 1 = Disease")
plt.ylabel("Amount")
plt.legend(["True","False"])
plt.xticks(rotation=0)


# We can see that Fasting Blood Sugar is not so realted to Heart disease

# #### Comparing Chest pain types and resting blood pressure with respect to gender

# In[22]:


sns.catplot(x="Chest_Pain_Type", y="Resting_Blood_Pressure",hue="Sex",kind ="swarm" ,data=df,palette="Dark2", height=7)
plt.xlabel('chest pain types')
plt.ylabel('Blood Pressure Rate')


# Catplot shows that Males with higher blood pressure have more chances of heart disease compared to females

# ### Modelling

# #### Imports

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# #### Train & Test Split

# In[58]:


X=df.drop('Heart_Disease',axis=1)
Y=df['Heart_Disease']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# #### RandomForestClassifier

# In[59]:


rf=RandomForestClassifier()


# In[60]:


rf.fit(X_train,Y_train)


# In[61]:


rf.score(X_train,Y_train)


# In[62]:


# Predict test variables
Y_pred = rf.predict(X_test)


# In[63]:


# Accuracy
rf.score(X_test,Y_test)


# In[64]:


# Comparing test and prediction for first 10 values
diffTable = pd.DataFrame({'Actual-Value': Y_test, 'Predicted-Value':Y_pred})
diffTable.head(10)


# 8 out of 10 predictions are correct, which equals to the 80% accuracy

# #### Model Evaluation

# In[51]:


# Plotting Confusion matrix
sns.set(font_scale=1.5)
fig,ax=plt.subplots(figsize=(3,3))
ax=sns.heatmap(confusion_matrix(Y_test,Y_pred),annot=True,cbar=False)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")


# In[42]:


# Classification report
print(classification_report(Y_test,Y_pred))


# In[ ]:




