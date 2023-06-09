import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
cancer_data=pd.read_csv('F:\major project\lungs cancer\survey_lung_cancer.csv')
cancer_data=pd.get_dummies(cancer_data,drop_first=True)
x=cancer_data.drop(columns=['LUNG_CANCER_YES'],axis=1)
y=cancer_data['LUNG_CANCER_YES']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)