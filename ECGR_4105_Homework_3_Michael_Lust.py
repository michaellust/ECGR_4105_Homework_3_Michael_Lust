#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Michael Lust : 801094861
#ECGR 4105 Intro to Machine Learning
#November 10, 2021
#Homework 3


# In[2]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_breast_cancer 


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer_data = cancer.data
cancer_data.shape


# In[5]:


cancer_input = pd.DataFrame(cancer_data)
cancer_input.head()


# In[6]:


#Organizing breast cancer data
cancer_labels = cancer.target
cancer_labels.shape


# In[7]:


labels = np.reshape(cancer_labels,(569,1))
final_cancer_data = np.concatenate([cancer_data,labels],axis=1)


# In[8]:


final_cancer_data.shape


# In[9]:


cancer_dataset = pd.DataFrame(final_cancer_data)
features = cancer.feature_names
features


# In[10]:


#Adding labeling to the data indexes
features_labels = np.append(features,'label') 
cancer_dataset.columns = features_labels 
cancer_dataset.head()


# In[11]:


#Classifying outcome as label to be Benign = 0 and Malignant = 1
cancer_dataset['label'].replace(0, 'Benign',inplace=True) 
cancer_dataset['label'].replace(1, 'Malignant',inplace=True) 


# In[12]:


cancer_dataset.head()


# In[13]:


cancer_dataset.tail()


# In[14]:


#Our Data set will consider Independent variables (X1-X30) and Label as Dependent (Y). 
X = cancer_dataset.iloc[:,0:30].values
Y = cancer_dataset.iloc[:,30].values
#see here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.
X[0:2] #Shows the Array


# In[15]:


Y[0:30] #Shows the outcome of either Benign or Malignant


# In[16]:


#Now we’ll split our Data set into Training Data and Test Data.
#Logistic model and Test data will be used to validate our model. We’ll use Sklearn.
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size =0.8, test_size = 0.2, random_state = 1)
X_train.shape


# In[17]:


#Problem 1 Logistic Regression Model


# In[18]:


#Now we’ll do feature scaling to scale our data between -1 and 1 using standardization.
#Here Scaling is important because there is a significant difference between explanatory variables.
from sklearn.preprocessing import StandardScaler 
scalar_X = StandardScaler()
X_train = scalar_X.fit_transform(X_train) #New X_train is scaled
X_test = scalar_X.transform(X_test) #New X_test is scaled


# In[19]:


X_train[0:2] #Checking the scaling of training set


# In[20]:


X_test[0:2] #Checking the scaling of testing set


# In[21]:


#Import LogisticRegression from sklearn.linear_model
#Make an instance classifier of the object LogisticRegression and give random_state = 0 
from sklearn.linear_model import LogisticRegression
classifier_L = LogisticRegression(random_state=0)
classifier_L.fit(X_train,Y_train)


# In[22]:


Y_pred = classifier_L.predict(X_test)


# In[23]:


Y_pred[0:29] #Checking Outcome from testing set


# In[24]:


#Using Confusion matrix representing binary classifiers so we can get accuracy of our model.
from sklearn.metrics import confusion_matrix 
cnf_matrix = confusion_matrix(Y_test,Y_pred)
cnf_matrix


# In[25]:


#We are evaluating the model using model evaluation metrics for accuracy.
#Accuracy over loss is what needs to be plotted
#Prediction and recall do not function with Y values or outcomes as string values.
from sklearn import metrics
Acc_score = metrics.accuracy_score(Y_test,Y_pred)
print("Accuracy:", Acc_score)


# In[26]:


#Plotting the confusion matrix:
import seaborn as sns 
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[27]:


#Plotting the loss vs. Number of Iterations for all X explanatory values combined does
#not seem doable with the logistic regression library. We will have to implement our own
#logistic regression function like we did for linear regression in problem 1.
#Logistic regression uses a log function to classify data points and will not suite 
#the training model for this dataset.


# In[28]:


#Problem 2 PCA Feature Extraction


# In[29]:


#Switching Classification of label from string to integer to Classification Accuracy, Precision, and Recall
cancer_dataset['label'].replace('Benign', 0, inplace=True) 
cancer_dataset['label'].replace('Malignant', 1, inplace=True) 


# In[30]:


#Our Data set will consider Independent variables (X1-X30) and Label as Dependent (Y). 
X = cancer_dataset.iloc[:,0:30].values
Y = cancer_dataset.iloc[:,30].values
#see here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.
Y[0:10]


# In[31]:


cancer_dataset.head()


# In[32]:


#Doing this scaling instead
#Now we’ll do feature scaling to scale our data between -1 and 1 using standardization.
#Here Scaling is important because there is a significant difference between explanatory variables. 
scalar_X = StandardScaler()
X = scalar_X.fit_transform(X) #New X is scaled
X[0:2]


# In[33]:


#Scaling using StandardScalar and separting features appropiately to run PCA
#features = all the feature names from before.
# Separating out the features 
#X = cancer_dataset.loc[:, features].values 
# Separating out the target 
#Y = cancer_dataset.loc[:,['label']].values 
# Standardizing the features 
#X = StandardScaler().fit_transform(X) 


# In[34]:


#Using Principle Component Analysis with specific N number of independent trainings
#Setting up function to plot precision, recall, and accuracy, over different K values.
#This involves creating a matrix to iterate over for the three results.
from sklearn.decomposition import PCA 
pcaDF = []

for i in range(29): #n_components must be between 0 and min(n_samples, n_features) = 30 with svd_solver='full'
    n_components = i + 2 #at least one array or dtype is required
    pca = PCA(n_components) #Iterating through all n values until reaching K.
    #principalComponents = pca.fit_transform(X_train)
    #-----------------------------------------------------------#
    principalComponents_train = pca.fit_transform(X)
    #------------------------------------------------------------#
    principalDf = pd.DataFrame(data = principalComponents_train, columns = range(n_components))
                 #, columns = ['principal component 1', 'principal component 2']) 
    finalDf = pd.concat([principalDf, cancer_dataset[['label']]], axis = 1)
    Y_principal = finalDf.iloc[:,[n_components]].values
    X_principal = finalDf.iloc[:,0:(n_components - 1)] 
    
    #Doing Training and Test Split to evaluate accuracy.
    np.random.seed(0)
    X_train, X_test, Y_train, Y_test = train_test_split(X_principal,Y_principal,train_size =0.8,test_size = 0.2,random_state = 1)
    
    #Using Logistic Regression for Classification
    classifier_L = LogisticRegression(random_state = 0)
    classifier_L.fit(X_train, np.ravel(Y_train)) #Changed the shape of y to (n_samples, ) 
    
    Y_pred1 = classifier_L.predict(X_test)
    Y_pred1[0:9]
    
    #Using Confusion matrix representing binary classifiers so we can get accuracy of our model.
    cnf_matrix = confusion_matrix(Y_test,Y_pred1)
    cnf_matrix
    
    #We are evaluating the model using model evaluation metrics for accuracy.
    #Accuracy over loss is what needs to be plotted
    from sklearn import metrics
    Acc_score1 = metrics.accuracy_score(Y_test,Y_pred1)
    print("Accuracy:", Acc_score1)
    Prec_score1 = metrics.precision_score(Y_test,Y_pred1)
    print("Precision:", Prec_score1)
    Rec_score1 = metrics.recall_score(Y_test,Y_pred1)
    print("Recall:", Rec_score1)
    
    pcaDF.append([n_components,Acc_score1,Prec_score1,Rec_score1])
print(pcaDF)
    
    


# In[35]:


#May not be needed
'''
fig = plt.figure(figsize = (8,8)) 
ax = fig.add_subplot(1,1,1)  
ax.set_xlabel('Principal Component 1', fontsize = 15) 
ax.set_ylabel('Principal Component 2', fontsize = 15) 
ax.set_title('2 component PCA', fontsize = 20) 
targets = ['Malignant', 'Benign'] 
colors = ['r', 'b'] 
for target, color in zip(targets,colors): 
    indicesToKeep = finalDf['label'] == target 
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'] 
               , finalDf.loc[indicesToKeep, 'principal component 2'] 
               , c = color 
               , s = 50) 
ax.legend(targets) 
ax.grid() 
'''


# In[36]:


#Plotting classification accuracy, precision, and recall over different numbers of Ks


# In[37]:


#function to get specific items from lists
def Parse(Value, i):
    return [item[i] for item in Value]


# In[38]:


N_to_K = Parse(pcaDF,0)
Acc_Vals = Parse(pcaDF,1)
Prec_Vals = Parse(pcaDF,2)
Recall_Vals = Parse(pcaDF,3)

plt.plot(N_to_K, Acc_Vals, 'r', label = 'Accuracy',linewidth=3)
plt.plot(N_to_K, Prec_Vals, 'g', label = 'Precision', linewidth=3)
plt.plot(N_to_K, Recall_Vals,  'b', label = 'Recall', linewidth=3)

plt.title("Results over different K values for PCA")
plt.xlabel("N Value")
plt.ylabel("Results from Accuracy, Precision, and Recall")
plt.legend(loc = 'lower right')
plt.show()


# In[39]:


#Problem 3 Using LDA feature extraction and Naive Bays Classifier for training


# In[40]:


#Switching Classification of label from string to integer to run Naive Bays
cancer_dataset['label'].replace('Benign', 0, inplace=True) 
cancer_dataset['label'].replace('Malignant', 1, inplace=True) 


# In[41]:


#Our Data set will consider Independent variables (X1-X30) and Label as Dependent (Y). 
X = cancer_dataset.iloc[:,0:30].values
Y = cancer_dataset.iloc[:,30].values
#see here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.
Y[0:10]


# In[42]:


#Now we’ll do feature scaling to scale our data between -1 and 1 using standardization.
#Here Scaling is important because there is a significant difference between explanatory variables. 
scalar_X = StandardScaler()
X = scalar_X.fit_transform(X) #New X_train is scaled
X[0:2]


# In[43]:


#Fitting LDA to breast cancer dataset: 
#Our dataset only has 2 classes. Therefore value of N can only be N-1, 1 for this instance:
lda = LinearDiscriminantAnalysis(n_components = 1) #n_components cannot be larger than min(n_features, n_classes - 1).

#lda_t = lda.fit_transform(X_train,Y_train) 
lda_train = lda.fit_transform(X,Y)
#lda_test = lda.transform(X_test,Y_test)
#Number of components (<= min(n_classes - 1, n_features)) for dimensionality reduction.


# In[44]:


#Doing training and test split
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(lda_train,Y,train_size =0.8, test_size = 0.2, random_state = 1)


# In[45]:


#Using Naive Gaussian Bays for classification
from sklearn.naive_bayes import GaussianNB
classifier_G = GaussianNB()
classifier_G.fit(X_train,Y_train)


# In[46]:


Y_pred = classifier_G.predict(X_test)
Y_pred


# In[47]:


conf_matrix = confusion_matrix(Y_test,Y_pred)
conf_matrix


# In[48]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
print("Accuracy:",metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))
print("Recall:",metrics.recall_score(Y_test,Y_pred))


# In[49]:


#Plotting the confustion Matrix
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[50]:


#Cannot plot classification accuracy, precision, and recall over different numbers of N given N can only equal 1.


# In[51]:


#Problem 4 Using LDA feature extraction and Logistic Regression for training


# In[52]:


#Our Data set will consider Independent variables (X1-X30) and Label as Dependent (Y). 
X = cancer_dataset.iloc[:,0:30].values
Y = cancer_dataset.iloc[:,30].values
#see here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.


# In[53]:


#Now we’ll do feature scaling to scale our data between -1 and 1 using standardization.
#Here Scaling is important because there is a significant difference between explanatory variables. 
scalar_X = StandardScaler()
X = scalar_X.fit_transform(X) #New X_train is scaled
X[0:2]


# In[54]:


#Fitting LDA to breast cancer dataset: 
lda = LinearDiscriminantAnalysis(n_components = 1) 
#lda_t = lda.fit_transform(X_train,Y_train)
lda_train = lda.fit_transform(X,Y)
#lda_test = lda.transform(X_test,Y_test)
#Number of components (<= min(n_classes - 1, n_features)) for dimensionality reduction.


# In[55]:


#Doing Training and Test Split to evaluate accuracy.
np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(lda_train,Y,train_size =0.8, test_size = 0.2, random_state = 1)


# In[56]:


#plt.xlabel('LD1') 
#plt.ylabel('LD2') 
#plt.scatter(lda_t[:,0],lda_t[:,1],c=y,cmap='rainbow',edgecolors='r')


# In[57]:


#Using Naive Logistic Regression for classification
classifier_L = LogisticRegression(random_state=0)
classifier_L.fit(X_train,Y_train)


# In[58]:


Y2_pred = classifier_L.predict(X_test)


# In[59]:


Y2_pred[0:9]


# In[60]:


#Using Confusion matrix representing binary classifiers so we can get accuracy of our model.
from sklearn.metrics import confusion_matrix 
cnf_matrix = confusion_matrix(Y_test,Y2_pred)
cnf_matrix


# In[61]:


#We are evaluating the model using model evaluation metrics for accuracy, precision, and recall.
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test,Y2_pred))
print("Precision:",metrics.precision_score(Y_test,Y2_pred))
print("Recall:",metrics.recall_score(Y_test,Y2_pred))


# In[62]:


#Plotting the confustion Matrix
class_names = [0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = "YlGnBu", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix',y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

