from packages import *
from data_loader import *
diabetes_svm = svm.SVC(kernel='linear')
diabetes_svm.fit(X_train, Y_train)
X_train_prediction = diabetes_svm.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy Score of Support Vector Machine is:', training_data_accuracy)
filename='diabetes_model.sav'
pickle.dump(diabetes_svm,open(filename,'wb'))
