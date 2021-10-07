#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction

# 1. [Import Libraries](#heading1)<br>
# 2. [Read Data](#heading2)<br>
# 3. [Data Cleaning & Feature Engineering](#heading3)<br>
# 4. [Exploratory Data Analysis](#heading4)<br>
# 5. [Model Building & Evaluation](#heading5)<br>
#   5.1 [Logistic Regression](#subheading1)<br>
#   5.2 [Gaussian Naive Bayes](#subheading2)<br>
#   5.3 [Linear Discriminant Analysis (LDA)](#subheading3)<br>
#   5.4 [k Nearest Neighbors (kNN)](#subheading4)<br>
#   5.5 [Support Vector Machine (SVM)](#subheading5)<br>
#   5.6 [Decision Tree](#subheading6)<br>
#   5.7 [Random Forest](#subheading7)<br>
#   5.8 [XGBoost](#subheading8)<br>
#   5.9 [Model Stacking](#subheading9)<br>
#   5.10 [Result Comparison](#subheading10)<br>
# 6. [Conclusion](#heading6)<br>

# ## 1. Import Libraries <a id="heading1"></a>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb


# In[2]:


# Set seed value for reproducing the same results
seed = 101


# ## 2. Read Data <a id="heading2"></a>

# In[3]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[4]:


# Train data preview
train_data.head()


# In[5]:


# Test data preview
test_data.head()


# We can see that the 'Survived' column is missing in the test set. We have to predict that label for each passenger in the test data.

# In[6]:


# Summary of train data
train_data.info()


# In[7]:


# Summary of test data
test_data.info()


# ## 3. Data Cleaning & Feature Engineering <a id="heading3"></a>

# In[8]:


# Train data descriptive statistics
train_data.describe()


# In[9]:


# Test data descriptive statistics
test_data.describe()


# For both train and test datasets, the statistics for 'Fare' column seem a bit strange. The minimum fare is 0 and the maximum is around 512, with 75% of values less than 31.5 and the mean being 35.6. We need to analyze this further to see if there are any outliers.
# 
# For this purpose, we can make use of a boxplot. It will help us understand the variation in the 'Fare' values by visually displaying the distribution of the data points.

# In[10]:


plt.subplots(figsize=(7, 5))
plt.boxplot(train_data['Fare'])
plt.title('Boxplot of Fare')
plt.show()


# It seems like there are a few extreme data points. Let's explore this further.

# In[11]:


# Retrieve rows with Fare greater than 500
train_data[train_data['Fare']>500]


# Since all of the passengers have the same ticket number, we can conclude that the fare was calculated for the entire group and not each individual. Hence, we will not discard these rows.
# 
# To standardize the fare calculation across all passengers in the dataset, the obvious step would be to divide fare by the number of people on the same ticket and get the individual fare. But factors such as reduced fares for children, missing values, etc., will further complicate things. Therefore, we will leave it as it is. For an in-depth understanding of the titanic dataset (particularly fare calculation), you can explore [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/).
# 
# Before we proceed further, we also need to analyze passengers who had 0 fare.

# In[12]:


# Retrieve rows with Fare equal to 0
train_data[train_data['Fare']==0]


# Again, it looks like there are no data errors; just some passengers who got a free ride for whatever reason (visit Encyclopedia Titanica if you're interested to find out why).
# 
# Next, we will check for missing values.

# In[13]:


# Number of missing values in each column in train data
train_data.isnull().sum()


# In[14]:


# Number of missing values in each column in test data
test_data.isnull().sum()


# First, let's deal with the missing 'Age' values. For that purpose, we will first extract title of each passenger from their name.

# In[15]:


# Function to extract title from passenger's name
def extract_title(df):
    title = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    return title


# In[16]:


# Count of each title in train data
train_data['Title'] = extract_title(train_data)
train_data['Title'].value_counts()


# In[17]:


# Count of each title in test data
test_data['Title'] = extract_title(test_data)
test_data['Title'].value_counts()


# Since there are many titles with very few counts, we will map them to main categories (titles that are more frequently occurring).

# In[18]:


# Function to map titles to main categories
def map_title(df):
    title_category = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
    }
    new_title = df['Title'].map(title_category)
    return new_title


# In[19]:


# Count of each title in train data after mapping
train_data['Title'] = map_title(train_data)
train_data['Title'].value_counts()


# In[20]:


# Count of each title in test data after mapping
test_data['Title'] = map_title(test_data)
test_data['Title'].value_counts()


# Now that we have extracted titles from names, we can group data by title and impute missing age values using the median age of each category. We will also group by 'Pclass' as it will help in accurately calculating the median age within each class.<br>
# Note: We are using median value instead of mean because extreme values (or outliers) have a lot more impact on mean than median.

# In[21]:


# Group train data by 'Pclass', 'Title' and calculate the median age
train_data.groupby(['Pclass', 'Title']).median()['Age']


# One thing to note here is that unlike the 'Master' title, there is no separate category for young female passengers. If we go back and look at the original dataset, we will realize that the 'Miss' title includes both young and adult females. We can somewhat solve this by identifying passengers with 'Miss' title having 1 or 2 value in the 'Parch' column. This way we can retrieve passengers who are most likely, young females (there's also a small chance that the retrieved passenger is a female adult because the 'Parch' column not only reveals the number of parents but also the number of children).

# In[22]:


# Function to identify passengers who have the title 'Miss' and, 1 or 2 value in the 'Parch' column
def is_young(df):
    young = []
    for index, value in df['Parch'].items():
        if ((df.loc[index, 'Title'] == 'Miss') and (value == 1 or value == 2)):
            young.append(1)
        else:
            young.append(0)
    return young


# In[23]:


# Group train data by 'Pclass', 'Title', 'Is_Young(Miss)' and calculate the median age
train_data['Is_Young(Miss)'] = is_young(train_data)
grouped_age = train_data.groupby(['Pclass', 'Title', 'Is_Young(Miss)']).median()['Age']
grouped_age


# This looks better as we can now guess the missing age values more accurately than before. We will apply this function to the test data as well.

# In[24]:


test_data['Is_Young(Miss)'] = is_young(test_data)


# Next, we will impute the missing age values according to the grouped data shown above.

# In[25]:


# Fill missing age values in train and test data
train_data.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)
train_data['Age'].fillna(grouped_age, inplace=True)
train_data.reset_index(drop=True, inplace=True)
test_data.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)
test_data['Age'].fillna(grouped_age, inplace=True)
test_data.reset_index(drop=True, inplace=True)


# A very important thing that needs to be addressed is that I've only used the train data to calculate the median ages for replacing missing values in both train and test datasets. Many people, especially those participating in data science competitions, use test data as well for preprocessing purposes. This may help people improve their model's test accuracy and rank higher in competitions, but it is considered a major mistake in real world applications (known as **data leakage**). Models built using this approach do not generalize too well to the new/unseen data and give results that are a lot poorer than expected. Hence, test data should never be used for data preprocessing and should only be used for testing purposes.
# 
# For replacing the missing 'Fare' value in test data, we will simply group the train data by 'Pclass' and repeat the same steps as above.

# In[26]:


# Group train data by 'Pclass' and calculate the median fare
grouped_fare = train_data.groupby('Pclass').median()['Fare']
grouped_fare


# In[27]:


# Fill the missing fare value in test data
test_data.set_index('Pclass', drop=False, inplace=True)
test_data['Fare'].fillna(grouped_fare, inplace=True)
test_data.reset_index(drop=True, inplace=True)


# Finally, we will drop all of the unnecessary rows and columns:
# * Name: We've extracted the information that we needed (i.e. Title) and don't need this column anymore
# * Cabin: Majority of the values are missing so we will drop the entire column
# * Embarked: Only 2 values are missing in train data so we can just remove those 2 entire rows
# * Ticket: Doesn't seem to provide any useful information so we will drop the entire column
# * Is_Young(Miss): Purpose of creating this column has been fulfilled and we don't need it anymore

# In[28]:


# Drop unnecessary rows and columns
train_data.drop(columns=['Name', 'Cabin', 'Ticket', 'Is_Young(Miss)'], inplace=True)
test_data.drop(columns=['Name', 'Cabin', 'Ticket', 'Is_Young(Miss)'], inplace=True)
train_data.dropna(subset=['Embarked'], inplace=True)


# It is always good to verify that there are no remaining missing values.

# In[29]:


# Missing values in train data after data cleaning
train_data.isnull().sum()


# In[30]:


# Missing values in test data after data cleaning
test_data.isnull().sum()


# ## 4. Exploratory Data Analysis <a id="heading4"></a>

# In this section, we will try to find some interesting insights using visual methods.
# 
# First, we will look at the class distribtuion.

# In[31]:


plt.subplots(figsize=(7, 5))
sns.countplot(x='Survived', data=train_data)
plt.title('Class Distribution')
plt.show()


# We can clearly see that the classes are slightly imbalanced since majority of the passengers did not survive. In scenarios like this, the same ratio is expected in test data so we don't need to worry about the imbalanced classes.
# 
# Next, let's find out the ratio of survivors with respect to other variables (i.e. 'Sex', 'Pclass', 'Embarked', 'Title').

# In[32]:


plt.subplots(figsize=(7, 5))
sns.barplot(x='Sex', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on sex')
plt.show()


# In[33]:


plt.subplots(figsize=(7, 5))
sns.barplot(x='Pclass', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on ticket class')
plt.show()


# In[34]:


plt.subplots(figsize=(7, 5))
sns.barplot(x='Embarked', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on port of embarkation')
plt.show()


# In[35]:


plt.subplots(figsize=(7, 5))
sns.barplot(x='Title', y='Survived', data=train_data, ci=None)
plt.title('Ratio of survivors based on title')
plt.show()


# Based on these visualizations, we can conclude the following:
# * Females had a way higher survival rate than males
# * Lower ticket class (with 3 being the lowest) means less chance of survival
# * Passengers who embarked from port 'C' had slightly more chances of survival
# * Passengers with the title 'Mr' and 'Officer' had really low chances of survival as compared to other passengers
# 
# Note: The accuracy of these findings also depends on other factors such as the frequency distribution within each categorical variable. For example, if there is only 1 female in the entire dataset and she survived, then the survival rate of females will be 100% which cannot be considered a concrete finding. Hence, depending on the type of problem being solved, further data analysis should be done if required.
# 
# Next, we will compute the pairwise correlation of different variables, focusing mainly on how different features correlate with the target variable 'Survived'. But first, we need to convert all of the categorical variables into numeric data type.
# 
# To convert 'Sex' variable into numeric format, we will simply encode male with 1 and female with 0.

# In[36]:


# Encode 'Sex' variable values
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
test_data['Sex'] = le.transform(test_data['Sex'])


# For 'Embarked' and 'Title' variables, we will use dummy variables to represent different values.

# In[37]:


# Convert 'Embarked' and 'Title' into dummy variables
train_data = pd.get_dummies(train_data, columns=['Embarked', 'Title'])
test_data = pd.get_dummies(test_data, columns=['Embarked', 'Title'])


# This is how the dataset looks like after conversion:

# In[38]:


train_data.head()


# Finally, we can calculate the correlation.

# In[39]:


# Pairwise correlation of columns
corr = train_data.corr()
corr


# Let's convert this into a visualization for better comprehension. 

# In[40]:


# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 8))

# Draw the heatmap with the mask
sns.heatmap(corr, mask=mask, cmap='RdBu_r', linewidths=.5, cbar_kws={'shrink': .7})
plt.show()


# If we just focus on the 'Survived' variable, we will notice that:
# * It has a comparatively strong negative correlation with 'Pclass', 'Sex' and 'Title_Mr'
# * It has a comparatively strong positive correlation with 'Fare', 'Embarked_C', 'Title_Miss' and 'Title_Mrs'

# ## 5. Model Building & Evaluation <a id="heading5"></a>

# Before we can start building the machine learning models, we need to apply feature scaling to standardize the independent variables within a particular range. This is required because some machine learning algorithms (such as kNN) tend to give more weightage to features with high magnitudes than features with low magnitudes, regardless of the unit of the values. To bring all features to the same level of magnitudes, we need to apply feature scaling.
# 
# In this case, we will use the MinMaxScaler to scale each feature to a (0, 1) range.

# In[41]:


# Apply feature scaling using MinMaxScaler
scaler = MinMaxScaler()
train_data.iloc[:, 2:] = scaler.fit_transform(train_data.iloc[:, 2:])
test_data.iloc[:, 1:] = scaler.transform(test_data.iloc[:, 1:])


# This is how the dataset looks like after feature scaling (remember, we only need to scale predictor variables):

# In[42]:


train_data.head()


# Next, we will split our train and test datasets with respect to predictor (X) and response (y) variables.

# In[43]:


X_train, X_test, y_train = train_data.iloc[:, 2:], test_data.iloc[:, 1:], train_data['Survived']


# The 'y_test' is not provided in this dataset. For getting the test scores, we will have to submit our predictions online. To make the entire process a bit smoother, we will write a function that takes in model predictions and generates a file in the required format to submit online.

# In[44]:


# Function to generate submission file to get test score
def submission(preds):
    test_data['Survived'] = preds
    predictions = test_data[['PassengerId', 'Survived']]
    predictions.to_csv('submission.csv', index=False)


# Now, we can finally start building machine learning models to predict which of the passengers survived.

# ### 5.1 Logistic Regression <a id="subheading1"></a>

# Important parameters that we will tune:
# * penalty: Used to specify the norm used in the penalization
# * C: Inverse of regularization strength
# 
# For hyperparameter tuning, we will use grid search cross validation over the specified parameter values. We will repeat 5-fold cross validation 10 times so that we can further improve the model performance and reduce overfitting. This will lead to better results for test/unseen data.

# In[45]:


# Classification model
logreg = LogisticRegression()

# Parameters to tune
params = [{'penalty': ['l1', 'l2', 'elasticnet', 'none'],
           'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
lr_clf = GridSearchCV(logreg, params, cv=cv, n_jobs=-1)
lr_clf.fit(X_train, y_train)


# In[46]:


# Best parameters
lr_clf.best_params_


# In[47]:


# Train score
lr_clf.best_score_


# The train accuracy is 82.7%.

# In[48]:


# Test score
y_preds = lr_clf.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 76.8%

# ### 5.2 Gaussian Naive Bayes <a id="subheading2"></a>

# Using default parameters.

# In[49]:


# Classification model
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[50]:


# Test score
y_preds = gnb.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 75.1%.

# ### 5.3 Linear Discriminant Analysis (LDA) <a id="subheading3"></a>

# Using default parameters.

# In[51]:


# Classification model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)


# In[52]:


# Test score
y_preds = lda.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 77.5%.

# ### 5.4 k Nearest Neighbors (kNN) <a id="subheading4"></a>

# Important parameters that we will tune:
# * n_neighbors: Number of neighbors to use
# * p: For choosing between manhattan distance and euclidean distance metrics

# In[53]:


# Classification model
knn = KNeighborsClassifier()

# Parameters to tune
params = [{'n_neighbors': range(1, 21),
           'p': [1, 2]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
knn_clf = GridSearchCV(knn, params, cv=cv, n_jobs=-1)
knn_clf.fit(X_train, y_train)


# In[54]:


# Best parameters
knn_clf.best_params_


# In[55]:


# Train score
knn_clf.best_score_


# The train accuracy is 82.1%.

# In[56]:


# Test score
y_preds = knn_clf.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 77.3%.

# ### 5.5 Support Vector Machine (SVM) <a id="subheading5"></a>

# Important parameters that we will tune:
# * C: Penalty parameter for determining the trade-off between setting a larger margin and lowering misclassification
# * kernel: Specifies the kernel type to be used in the algorithm

# In[57]:


# Classification model
svm = SVC(max_iter=10000)

# Parameters to tune
params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
           'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
svm_clf = GridSearchCV(svm, params, cv=cv, n_jobs=-1)
svm_clf.fit(X_train, y_train)


# In[58]:


# Best parameters
svm_clf.best_params_


# In[59]:


# Train score
svm_clf.best_score_


# The train accuracy is 82.8%.

# In[60]:


# Test score
y_preds = svm_clf.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 77.8%.

# ### 5.6 Decision Tree <a id="subheading6"></a>

# Important parameters that we will tune:
# * max_depth: Maximum depth of the tree
# * min_samples_split: Minimum number of samples required to split an internal node
# * max_features: Number of features to consider when looking for the best split

# In[61]:


# Classification model
dt = DecisionTreeClassifier(random_state=seed)

# Parameters to tune
params = [{'max_depth': [5, 7, 10, None],
           'min_samples_split': [2, 5, 10],
           'max_features': ['sqrt', 5, 7, 10]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
dt_clf = GridSearchCV(dt, params, cv=cv, n_jobs=-1)
dt_clf.fit(X_train, y_train)


# In[62]:


# Best parameters
dt_clf.best_params_


# In[63]:


# Train score
dt_clf.best_score_


# The train accuracy is 81.6%.

# In[64]:


# Test score
y_preds = dt_clf.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 78%.

# ### 5.7 Random Forest <a id="subheading7"></a>

# Important parameters that we will tune:
# * n_estimators: Number of trees in the forest
# * max_depth: Maximum depth of the tree
# * min_samples_split: Minimum number of samples required to split an internal node
# * max_features: Number of features to consider when looking for the best split

# In[65]:


# Note: This cell will take a while to run depending on the available processing power

# Classification model
rf = RandomForestClassifier(random_state=seed)

# Parameters to tune
params = [{'n_estimators': range(50, 550, 50),
           'max_depth': [5, 7, 10, None],
           'min_samples_split': [2, 5, 10],
           'max_features': ['sqrt', 5, 7, 10]}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
rf_clf = GridSearchCV(rf, params, cv=cv, n_jobs=-1)
rf_clf.fit(X_train, y_train)


# In[66]:


# Best parameters
rf_clf.best_params_


# In[67]:


# Train score
rf_clf.best_score_


# The train accuracy is 83.7%.

# In[68]:


# Test score
y_preds = rf_clf.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 77%.

# ### 5.8 XGBoost <a id="subheading8"></a>

# Important parameters that we will tune:
# * max_depth: Maximum depth of the tree
# * learning_rate: Controls the contribution of each tree
# * n_estimators: Number of trees

# In[69]:


# Note: This cell will take a while to run depending on the available processing power

# Classification model
xgboost = xgb.XGBClassifier(random_state=seed)

# Parameters to tune
params = [{'max_depth': [3, 5, 10],
           'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
           'n_estimators': range(100, 1100, 100)}]

# Hyperparameter tuning using GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
xgb_clf = GridSearchCV(xgboost, params, cv=cv, n_jobs=-1)
xgb_clf.fit(X_train, y_train)


# In[70]:


# Best parameters
xgb_clf.best_params_


# In[71]:


# Train score
xgb_clf.best_score_


# The train accuracy is 82.9%.

# In[72]:


# Test score
y_preds = xgb_clf.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 76.8%.

# ### 5.9 Model Stacking <a id="subheading9"></a>

# In this part, we will stack all of our best performing models using the stacking classifier. Predictions generated by various models will be optimally combined to form a new set of predictions. (Note: The new predictions may not always give better result than the individual models).
# 
# Using default parameters.

# In[73]:


# Models that we will input to stacking classifier
base_estimators = list()
base_estimators.append(('lda', lda))
base_estimators.append(('knn', knn_clf.best_estimator_))
base_estimators.append(('svm', svm_clf.best_estimator_))
base_estimators.append(('dt', dt_clf.best_estimator_))
base_estimators.append(('rf', rf_clf.best_estimator_))

# Stacking classifier
stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
stacking_clf.fit(X_train, y_train)


# In[74]:


# Test score
y_preds = stacking_clf.predict(X_test)
submission(y_preds)


# After submission, the test accuracy is found to be 78%.

# ### 5.10 Result Comparison <a id="subheading10"></a>

# | Model | Train Accuracy (%) | Test Accuracy (%) |
# | ----- | ------------------ | ----------------- |
# | Logistic Regression | 82.7 | 76.8 |
# | Gaussian Naive Bayes | N/A | 75.1 |
# | Linear Discriminant Analysis | N/A | 77.5 |
# | k Nearest Neighbors | 82.1 | 77.3 |
# | Support Vector Machine | 82.8| 77.8 |
# | Decision Tree | 81.6 | 78 |
# | Random Forest | 83.7 | 77 |
# | XGBoost | 82.9 | 76.8 |
# | Model Stacking | N/A | 78 |

# Looking at the above table, we can observe the following:
# * Random Forest gave the highest train accuracy of 83.7%
# * Decision Tree and Stacking Classifier performed best for test/unseen data with an accuracy of 78%
# * Most of the models performed really similar in terms of test accuracy
# * Due to the small dataset size, all models have (slightly) overfitted the train data, giving lower test scores than expected

# ## 6. Conclusion <a id="heading6"></a>

# This notebook gave a brief overview of how different steps are performed in a data science project life cycle. We started by reading in the dataset, preprocessing it, exploring it to find useful insights, and finally built various machine learning models and evaluated them. The main objective of this project was to analyze the titanic dataset and predict whether a passenger will survive or not, based on various input features. To further build and improve upon this project, a lot of techniques could be tried.
# 
# Innovative ways of feature engineering like combining the 'SibSp' and 'Parch' features, or applying different data preprocessing methods such as binning the 'Age' column could be tried to help improve the overall performance. One technique that will surely improve the scores is to further hypertune the models. Due to limited time and processing power available, we only performed grid search over a few combinations of paramters' values (we also skipped many parameters and used their default value). The extra time spent on tuning the parameters usually leads to better results.
# 
# Additionally, there are other options for trying and improving the prediction accuracy such as applying feature selection techniques or building deep learning models (e.g. neural networks). Part of a job of data scientists is to be creative, keep experimenting and try figuring out new ways of improving upon their work. The 'titanic survival prediction' task is no exception.
