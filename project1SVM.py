import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# Data collection and preprocessing
heart_data=pd.read_csv(r'C:\Users\Aju Pradhan\Desktop\all_together_project\heart.csv')

X=heart_data.drop(columns='target', axis=1)
Y=heart_data['target']

scalar = StandardScaler()
scalar.fit(X)

standardised_data= scalar.transform(X)
X=standardised_data
Y=heart_data['target']

# print(X)
# print(Y)

# splitting the data into training data and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

# print(X.shape, X_train.shape, X_test,shape)

# create a svm classifier
clf=svm.SVC(kernel='linear')

# train the model using training set
clf.fit(X_train, Y_train)

# model Evaluation
# Accuracy Score

# Accuracy on training data
X_train_prediction=clf.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data :', training_data_accuracy)


#  Accuracy on test data
X_test_prediction=clf.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data :',test_data_accuracy)


filename='heartdisease_model.sav'
pickle.dump(clf , open(filename, 'wb'))

# =============================================================================
# input_data=(52,1,1,134,201,0,1,158,0,0.8,2,1,2)
# 
# # change the input data to a numpy array
# input_data_as_numpy_array= np.asarray(input_data)
# 
# # reshape the numpy array as we are predicting for only one instance
# input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
# 
# prediction= clf.predict(input_data_reshaped)
# print(prediction)
# 
# if(prediction[0]==0):
#     print("The person doesn't have haert disease")# Building a predictive system
# 
# 
# else:
#         print("the person has heart disease")
# =============================================================================

