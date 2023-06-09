from packages import *
from data_loader import *
model_logistic = LogisticRegression()
#model_logisticRegression=LogisticRegression()
model_logistic.fit(X_train, Y_train)
X_train_prediction = model_logistic.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model_logistic.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy Score of Logistic Regression model is:', training_data_accuracy)
filename='diabetes_model(Logistic Regression).sav'
pickle.dump(model_logistic,open(filename,'wb'))
loaded_model=pickle.load(open('diabetes_model(Logistic Regression).sav','rb'))
