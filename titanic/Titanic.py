#!/usr/bin/env python
# coding: utf-8

# ## Importing Stuff

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from time import time


# ## Viewing the dataset

# Let's take a view into the dataset itself.

# In[2]:


data_raw = pd.read_csv("datasets/titanic_train.csv", index_col='PassengerId')
data_validate = pd.read_csv("datasets/titanic_test.csv", index_col='PassengerId')
data_raw.sample(10)


# In[3]:


data_raw.info()


# In[4]:


data_raw.isnull().sum()


# In[5]:


data_raw.describe(include='all')


# In[6]:


data_raw['Sex'].value_counts()


# In[7]:


data_raw['Embarked'].value_counts()


# ## Cleaning and Wrangling the Data

# We'll make a copy of the raw data and put it in a list along with the validation set. We can later separate it into training and testing data.

# In[8]:


data_copy = data_raw.copy(deep=True)
data_cleaner = [data_copy, data_validate]


# We see that there are 891 entries in the dataset and 12 columns including the PassengerId as the index.
# 
# Of the 891 entries for Cabin 687 entries in total are null. This means that there isn't much we can do with the information about the cabin.
# 
# In addition, both the Ticket and Fare columns are more or less random. Furthermore, PassengerId is only a unique identifier and will not affect our model.
# 
# While it is possible to separate the Name into titles alone, I believe it is not needed.
# 
# So all of them are dropped.

# 
# We note that there 177 entries for Age do not exist. Instead of deleting these entries completely, we shall instead fill these age columns with the median age. We choose median over mean because there are both babies(Age is a fraction less than one) and very old people as well which might skew the value of mean.

# In the case of the port of Embarkation, we see that only 2 values are null. We will use the mode of this column to fill in these values.

# In[9]:


for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset.drop(['Cabin', 'Ticket', 'Fare', 'Name'], axis=1, inplace = True)


# Now SibSp and Parch are described as follows:
# 
#     sibsp: The dataset defines family relations in this way...
#     Sibling = brother, sister, stepbrother, stepsister
#     Spouse = husband, wife (mistresses and fiancés were ignored)
# 
#     parch: The dataset defines family relations in this way...
#     Parent = mother, father
#     Child = daughter, son, stepdaughter, stepson
#     Some children travelled only with a nanny, therefore parch=0 for them.
#     
# We can instead create a new feature 'FamilySize' by adding SibSp and Parch and 1(For the person themself). We will also create another feature 'IsAlone' for the people who travelled alone. And then we may remove the SibSp and Parch columns.

# In[10]:


for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # We set IsAlone to 1/True for everyone and then change it to 0/False depending on their FamilySize.
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    dataset.drop(['SibSp', 'Parch'], axis=1, inplace = True)


# In[11]:


data_cleaner[0].head()


# We know from our preliminary analysis that the Sex is either male or female. We also know the age and port of embarkation for all passengers. Let us set male = 0 and female = 1. Also, we can set the port such that C = 0, Q = 1, S = 2. We shall leave the age as it is

# In[12]:


for dataset in data_cleaner:
    dataset['Sex'].loc[dataset['Sex'] == 'male'] = 0
    dataset['Sex'].loc[dataset['Sex'] == 'female'] = 1
    dataset['Embarked'].loc[dataset['Embarked'] == 'C'] = 0
    dataset['Embarked'].loc[dataset['Embarked'] == 'Q'] = 1
    dataset['Embarked'].loc[dataset['Embarked'] == 'S'] = 2


# In[13]:


data_cleaner[0].head()


# ## Splitting up the data

# We can now split the data into the labels and features.

# In[14]:


data_clean, data_validate = data_cleaner
data_labels = data_clean['Survived']
data_features = data_clean.drop('Survived', axis=1)


# Splitting up the labels and features into training and testing sets.

# In[15]:


features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels,
                                                                            test_size=0.2, random_state=42)


# Taking a look at our testing, training and validating data

# ##### Training Data

# In[16]:


features_train.head()


# In[17]:


labels_train.head()


# ##### Testing Data

# In[18]:


features_test.head()


# In[19]:


labels_test.head()


# ##### Validation Data

# In[20]:


data_validate.head()


# ## Applying Naive Bayes

# In[21]:


nb_classifier = GaussianNB()


# In[22]:


t0 = time()
nb_classifier.fit(features_train, labels_train)
print("Training Time: ", time()-t0, "s.", sep='')


# In[23]:


t1 = time()
nb_pred = nb_classifier.predict(features_test)
print("Testing Time: ", time()-t1, "s.", sep='')


# In[24]:


print("Accuracy: ", accuracy_score(labels_test, nb_pred), ".", sep='')


# ~79% accuracy on our testing data. Now we just predict for the given data and save it to a file.

# ## Using a Decision Tree

# In[25]:


dt_classifier = tree.DecisionTreeClassifier(min_samples_split=40)


# In[26]:


t0 = time()
dt_classifier.fit(features_train, labels_train)
print("Training Time: ", round(time() - t0), "s")


# In[27]:


t1 = time()
dt_prediction = dt_classifier.predict(features_test)
print("Prediction Time: ", round(time() - t1), "s")


# In[28]:


print(accuracy_score(labels_test, dt_prediction))


# In[29]:


features_test.head()


# In[35]:


dt_classifier.predict(features_test.head())


# In[37]:


labels_test[:5]


# ## Running the algorithms on the validation set

# In[39]:


final = dt_classifier.predict(data_validate)


# In[40]:


sample = pd.read_csv("datasets/titanic_sample.csv", index_col='PassengerId')
sample['Survived'] = final
sample.to_csv("datasets/titanic_output.csv", )


# # Kaggle Score: 0.75119

# <img src="result.png">

# That's about it. I know I didn't do any of the fancy graphs and explore the data further, but seeing as I started learning about 2 days ago, I'll take it. I might update this in the future but most likely it'll remain the same. If I do revisit this, I will probably go more indepth and try out different models.
