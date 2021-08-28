#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("breast.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df['Unnamed: 32']


# In[9]:


df = df.drop("Unnamed: 32", axis=1)


# In[10]:


df.head()


# In[11]:


df.drop('id', axis=1, inplace=True)


# In[12]:


l=list(df.columns)
l


# In[13]:


df.head(2)


# In[14]:


df['diagnosis'].unique()


# In[15]:


sns.countplot(df['diagnosis'], label="Count",);


# In[16]:


df['diagnosis'].value_counts()


# In[17]:


df.shape


# # Explore The Data

# In[18]:


df.describe()


# In[19]:


#correlation plot
corr = df.corr()
corr


# In[20]:


corr.shape


# In[21]:


plt.figure(figsize=(10,10))
sns.heatmap(corr);


# In[22]:


#sns.pairplot(df)
#plt.show()


# In[23]:


df.head()


# In[24]:


df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
df.to_csv('tits.csv')
df.head()


# In[25]:


df['diagnosis'].unique()


# In[26]:


X=df.drop('diagnosis',axis=1)
X.head()


# In[27]:


y=df['diagnosis']
y.head()


# # Train Test Split

# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[29]:


print(X_train.shape ,X_test.shape)
print(y_train.shape, y_test.shape)


# In[30]:


X_train.head(1)


# In[31]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[32]:


X_train


# In[33]:


X_test


# # Machine learning Models

# ## Logistic Regression

# In[34]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(random_state = 5)
lr.fit(X_train,y_train)


# In[35]:


y_pred = lr.predict(X_test)
y_pred


# In[36]:


y_test


# In[37]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[38]:


lr_acc = accuracy_score(y_test, y_pred)


# In[39]:


results = pd.DataFrame()
results


# In[40]:


tempResult = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat([results, tempResult])
results


# # Decision Tree Classifier

# In[41]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[42]:


y_pred = dtc.predict(X_test)
y_pred


# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[44]:


dtc_acc = accuracy_score(y_test, y_pred)


# In[45]:


tempResult = pd.DataFrame({'Algorithm':['Decision Tree Classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat([results, tempResult])
results = results[['Algorithm','Accuracy']]
results


# # Random Forest Classifier

# In[46]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfc.fit(X_train, y_train)


# In[47]:


y_pred = rfc.predict(X_test)
y_pred


# In[48]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[49]:


rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)


# In[50]:


tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Support Vector Classifier 

# In[51]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)


# In[52]:


y_pred = svc.predict(X_test)
y_pred


# In[53]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[54]:


svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)


# In[55]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # KNN Classifier 

# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean', p = 2)
knn.fit(X_train, y_train)


# In[57]:


y_pred = knn.predict(X_test)
y_pred


# In[58]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[59]:


knn_acc = accuracy_score(y_test, y_pred)
print(knn_acc)


# In[60]:


tempResults = pd.DataFrame({'Algorithm':['K-Nearest-Neighbor Classification Method'], 'Accuracy':[knn_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Neive Bayes Classifier

# In[61]:


from sklearn.naive_bayes import GaussianNB
nbc = GaussianNB()
nbc.fit(X_train, y_train)


# In[62]:


y_pred = nbc.predict(X_test)
y_pred


# In[63]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[64]:


nbc_acc = accuracy_score(y_test, y_pred)
print(nbc_acc)


# In[65]:


tempResults = pd.DataFrame({'Algorithm':['Neive Bayes Classification Method'], 'Accuracy':[nbc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # ANN

# In[66]:


import tensorflow as tf


# In[67]:


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[68]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[69]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# In[70]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
type(y_pred)
y_pred = y_pred+0
y_pred = y_pred.reshape(len(y_pred))
y_pred


# In[71]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[72]:


ann_acc = accuracy_score(y_test, y_pred)
print(ann_acc)


# In[73]:


tempResults = pd.DataFrame({'Algorithm':['Artificial Neural Network Method'], 'Accuracy':[ann_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Stochastic Gradient Descent 

# In[74]:


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, y_train)


# In[75]:


y_pred = sgd.predict(X_test)
y_pred


# In[76]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[77]:


sgd_acc = accuracy_score(y_test, y_pred)
print(sgd_acc)


# In[78]:


tempResults = pd.DataFrame({'Algorithm':['Stochastic Gradient Descent Method'], 'Accuracy':[sgd_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Adaboost

# In[79]:


from sklearn.ensemble import AdaBoostClassifier
ab=AdaBoostClassifier(n_estimators=2500)
ab.fit(X_train, y_train)


# In[80]:


y_pred = ab.predict(X_test)
y_pred


# In[81]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[82]:


ab_acc = accuracy_score(y_test, y_pred)
print(ab_acc)


# In[83]:


tempResults = pd.DataFrame({'Algorithm':['AdaBoost Method'], 'Accuracy':[ab_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Multi Layer Neuron Classifier

# In[84]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=50)
mlp.fit(X_train, y_train)


# In[85]:


y_pred = mlp.predict(X_test)
y_pred


# In[86]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
print("Accuracy Score : ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred, digits=5))


# In[87]:


mlp_acc = accuracy_score(y_test, y_pred)
print(mlp_acc)


# In[88]:


tempResults = pd.DataFrame({'Algorithm':['Multi Layer Neuron Classification Method'], 'Accuracy':[mlp_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[89]:


results.to_csv('accuracy.csv')


# In[90]:


results.Accuracy.plot.bar()


# In[91]:


results['Accuracy'].value_counts()


# # Input The Values

# In[92]:


#input
l = [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,
     0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,
     0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]


# # Prediction By all the Classifiers

# In[93]:


print("Breast Cancer Detection by Logistic Regression Classifier : ",lr.predict(sc.transform([l])))
print("Breast Cancer Detection by Desicion Tree Classifier : ",dtc.predict(sc.transform([l])))
print("Breast Cancer Detection by Random Forest Classifier : ",rfc.predict(sc.transform([l])))
print("Breast Cancer Detection by Support Vector Machine Classifier : ",svc.predict(sc.transform([l])))
print("Breast Cancer Detection by K-Nearest-Neighbor Classifier : ",knn.predict(sc.transform([l])))
print("Breast Cancer Detection by Neive Based Classifier : ",nbc.predict(sc.transform([l])))
print("Breast Cancer Detection by Artificial Neural Network Classifier : ",ann.predict(sc.transform([l])))
print("Breast Cancer Detection by Stochastic Gradient Descent Classifier : ",sgd.predict(sc.transform([l])))
print("Breast Cancer Detection by AdaBoost Classifier : ",ab.predict(sc.transform([l])))
print("Breast Cancer Detection by Multi Layer Neuron Classifier : ",mlp.predict(sc.transform([l])))


# In[ ]:




