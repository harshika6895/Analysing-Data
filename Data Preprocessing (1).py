#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing
# 
# 
# Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.

# Steps in Data processing-
# Step 1 : Import the necessary libraries
# 

# In[27]:


import numpy as np
import pandas as pd 
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Step 2 : Load the dataset

# In[5]:


Pima_dataset = pd.read_csv('ML/diabetes.csv')
print(Pima_dataset.head())


# In[8]:


Pima_dataset.info()



# Step 3 : Statstical Analysis

# In[9]:


Pima_dataset.describe()


# Let's plot the boxplot for each coloumn 
# 
# Step 4 : Check the outliers 

# In[11]:


#Box plot
fig, axs = plt.subplots(9,1,dpi=95, figsize=(7,17))
i=0
for col in Pima_dataset.columns:
    axs[i].boxplot(Pima_dataset[col] , vert =False)
    axs[i].set_ylabel(col)
    i+=1
plt.show()


# Drop the outliers:

# In[19]:


#Identify the quartiles
q1 , q3 = np.percentile(Pima_dataset['Insulin'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = Pima_dataset[(Pima_dataset['Insulin'] >= lower_bound) & (Pima_dataset['Insulin'] <= upper_bound)] 

#Identify the quartiles
q1 , q3 = np.percentile(clean_data['Pregnancies'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = clean_data[(clean_data['Pregnancies']>=lower_bound) & (clean_data['Pregnancies']<=upper_bound)] 

#Identify the quartiles
q1 , q3 = np.percentile(clean_data['SkinThickness'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = clean_data[(clean_data['SkinThickness']>=lower_bound) & (clean_data['SkinThickness']<=upper_bound)]
                                     
#Identify the quartiles
q1 , q3 = np.percentile(clean_data['BloodPressure'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = clean_data[(clean_data['BloodPressure']>=lower_bound) & (clean_data['BloodPressure']<=upper_bound)] 

#Identify the quartiles
q1 , q3 = np.percentile(clean_data['Glucose'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = clean_data[(clean_data['Glucose']>=lower_bound) & (clean_data['Glucose']<=upper_bound)]

#Identify the quartiles
q1 , q3 = np.percentile(clean_data['BMI'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = clean_data[(clean_data['BMI']>=lower_bound) & (clean_data['BMI']<=upper_bound)]
                                    
#Identify the quartiles
q1 , q3 = np.percentile(clean_data['Age'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = clean_data[(clean_data['Age']>=lower_bound) & (clean_data['Age']<=upper_bound)]
                                     
#Identify the quartiles
q1 , q3 = np.percentile(clean_data['DiabetesPedigreeFunction'],[25,75])
# calculate the interquartile range 
iqr = q3-q1
#calculate the lower and upper bound
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)
#Drop the outliers 
clean_data = clean_data[(clean_data['DiabetesPedigreeFunction']>=lower_bound) & (clean_data['DiabetesPedigreeFunction']<=upper_bound)]





# Step 5 : Correlation (https://www.datacamp.com/tutorial/tutorial-datails-on-correlation)
# 
# Correlation is the statistical analysis of the relationship or dependency between two variables. Correlation allows us to study both the strength and direction of the relationship between two sets of variables.
# 
# First, it is a key component in data exploratory analysis
# Second, correlations have many real-world applications. They can help us answer questions, such as whether there is a link between democracy and economic growth, or whether the use of cars correlates to the level of air pollution.
# Finally, the study of correlation is critical in the field of machine learning. For example, some algorithms will not work properly if two or more variables are closely related, usually known as multicollinearity
# 
# we wanted to explore the correlation between all the pairs of variables, we could simply use the .corr() method directly to our DataFrame, which results again in a correlation matrix with the coefficient of all the pairs of variables.To extract the insights of our matrix in a more effective way, we could use a heatmap; a data visualization technique where each value is represented by a color, according to its intensity in a given scale.

# In[21]:


#correlation
corr = Pima_dataset.corr()
plt.figure(dpi = 130)
sns.heatmap(Pima_dataset.corr(), annot= True, fmt='.2f')
plt.show()


# In[22]:


corr['Outcome'].sort_values(ascending = False)


# In[24]:


#check outcome proportionality 

plt.pie(Pima_dataset.Outcome.value_counts(), labels = ['Diabetic','Non Diabetic'], autopct='%.f' , shadow=True)
plt.title('Outcome Proportionality')
plt.show()


# Step 6 : Separate Independent features and target variables 

# In[25]:


#separate into input and output
X = Pima_dataset.drop(columns= ['Outcome'])
Y= Pima_dataset.Outcome


# Step 7 : Normalization and Standardization
# 
# Normalization : MinMaxScaler scales the data so that each feature is in the range [0, 1]. 
# It works well when the features have different scales and the algorithm being used is sensitive to the scale of the features, such as k-nearest neighbors or neural networks.

# In[26]:


#Initialising the MinMaxScaler
Scaler = MinMaxScaler(feature_range = (0,1))
rescaledX = Scaler.fit_transform(X)
rescaledX[:5]


# Standardization (https://www.analyticsvidhya.com/blog/2022/10/understand-the-concept-of-standardization-in-machine-learning/)
# 
# : Standardization is a useful technique to transform attributes with a Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1.
# We can standardize data using scikit-learn with the StandardScaler class.
# It works well when the features have a normal distribution or when the algorithm being used is not sensitive to the scale of the features
# 

# In[42]:


# fit the scaler, it will learn the parameters 
Scaler = StandardScaler().fit(X)
#transform set
rescaledX = Scaler.transform(X)
rescaledX[:5]
Xscaled = pd.DataFrame(rescaledX , columns = X.columns)


# There will always be a debate about whether to do a train test split before doing standard scaling or not, but in my preference, it is essential to do this step before standardization as if we scale the whole data, then for our algorithm, there might be no testing data left which will eventually lead to overfitting condition. On the other hand now if we only scale our training data then we will still have the testing data which is unseen for the model.

# In[41]:


X.describe()


# In[43]:


Xscaled.describe()

