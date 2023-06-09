import for_data_loading
import data_loader
import project1SVM
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
# loading the saved models
model=pickle.load(open(r'C:\Users\Aju Pradhan\Desktop\all_together_project\model.sav','rb'))
model1=pickle.load(open(r'C:\Users\Aju Pradhan\Desktop\all_together_project\heartdisease_model.sav','rb'))
model2=pickle.load(open(r'C:\Users\Aju Pradhan\Desktop\all_together_project\diabetes_model.sav','rb'))
#   sidebar for navigate
with st.sidebar:
    selected=option_menu("disease prediction",['lung cancer','heart disease','diabetes','about us','help'],default_index=0)
# lung cancer part
if(selected=="lung cancer"):
    st.title("predict your lung cancer")

    age=st.number_input("enter your age")
    smoking=st.number_input("do you smoke ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    yellow_fingers=st.number_input("do you have yellowish fingers? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    anxiety=st.number_input("do you have anxiety ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    peer_pressure=st.number_input("do you have peer pressure ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    chronic_disease=st.number_input("do you have chronic disease ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    fatigue=st.number_input("do you have fatigue ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    allergy=st.number_input("do you allergy ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    wheezing=st.number_input("do you wheezing ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    alcohol_consuming=st.number_input("do you consume alcohol? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    coughing=st.number_input("do you having cough ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    shortness_of_breath=st.number_input("do you have shortness of breath ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    swallowing_difficulty=st.number_input("do you have swallowing difficulty ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    chest_pain=st.number_input("do you have chest pain ? if 'yes' enter '2' if no enter '1'",min_value=1,max_value=2,step=1)
    gender=st.number_input("what is your gender ? if 'male' enter '1' if 'female' enter '0'",min_value=0,max_value=1,step=1)

    input_data=(68,2,1,2,1,1,2,1,1,1,1,1,1,1,0)
    # changing input data to a numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    # reshape the numpy array
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    # we must standardized data because the data is trained in standardized form
    # std_data=for_data_loading.scalar.transform(input_data_reshaped)    
    input_data=(age,smoking,yellow_fingers,anxiety,peer_pressure,chronic_disease,fatigue,allergy,wheezing,alcohol_consuming,coughing,shortness_of_breath,swallowing_difficulty,chest_pain,gender)
    # input_data=(int(age),int(smoking),int(yellow_fingers),int(anxiety),int(peer_pressure),int(chronic_disease),int(fatigue),int(allergy),int(wheezing),int(alcohol_consuming),int(coughing),int(shortness_of_breath),int(swallowing_difficulty),int(chest_pain),int(gender))
    # changing input data to a numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    # reshape the numpy array
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    std_data=for_data_loading.scalar.transform(input_data_reshaped)
    # code for prediction
    lung_diagnosis=''
    cancer_prediction=[]
    # creating a button for prediction
    if st.button('lung cancer test result'):
        cancer_prediction=model.predict(std_data)

        if(cancer_prediction[0]==1):
            lung_diagnosis='the person seems to have lung cancer'
        else:
            lung_diagnosis='the person seems to does not have lung cancer'
    st.success(lung_diagnosis)            

# heart disease part
if (selected == 'heart disease'):
    
    #Page Title
    st.title('Heart disease prediction using ml')
    
    age1= st.text_input('Age of the person')
    sex = st.text_input('Sex of the person')
    cp = st.text_input('Chest Pain type') 
    trestbps = st.text_input('Resting Bool pressure') 
    chol = st.text_input('serum cholesterol in mg/dl')
    fbs = st.text_input('fasting blood sugar > 120 mg/dl')
    restecg = st.text_input('resting electrocardiographic results (values 0,1,2)')
    thalach= st.text_input('maximum heart rate achieved')
    exang = st.text_input('exercise induced angina')
    oldpeak = st.text_input('oldpeak = ST depression induced by exercise relative to rest')
    slope = st.text_input('the slope of the peak exercise ST segment')
    ca = st.text_input('number of major vessels (0-3) colored by flourosopy')
    thal = st.text_input('hal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    heart_diagnosis= " "
 
    if st.button("Heart disease test result"):
    # loaded_model=pickle.load(open('heartdisease_model.sav' , 'rb'))
    
    #put the data here
        input_data1=(age1,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        input_data_as_numpy_array1= np.asarray(input_data1)
        input_data_reshaped1=input_data_as_numpy_array1.reshape(1,-1)
        std_data1=project1SVM.scalar.transform(input_data_reshaped1)
        prediction1=model1.predict(std_data1)
        st.write([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        heart_diagnosis=prediction1
    
        if(heart_diagnosis[0]==1):
            st.success("the Person has heart disease")
        else:
            st.success("the Person does not have heart disease")
       
        st.success(heart_diagnosis)

# for diabetes
if (selected=='diabetes'):
    st.title('Diabetes Prediction')
    col1,col2 =st.columns(2)
    with col1:
        Pregnancies=st.text_input('Pregnancies')
    with col2:
        Glucose=st.text_input('Glucose')
    with col1:
        BloodPressure=st.text_input('Blood Pressure')
    with col2:
        SkinThickness=st.text_input('SkinThickness')
    with col1:
        Insulin=st.text_input('Insulin')
    with col2:
        BMI=st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function')
    with col2:
        Age=st.text_input('Age')
    
    
    diabetes_result=''
    if st.button("Diabetes test result"):
        
        # loaded_model=pickle.load(open('diabetes_model.sav','rb'))
        #Put the data here
        
        input_data2=(Pregnancies,Glucose,BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        input_data_as_numpy_array2=np.asarray(input_data2)
        input_data_reshaped2=input_data_as_numpy_array2.reshape(1,-1)
        std_data2=data_loader.scalar.transform(input_data_reshaped2)
        prediction2=model2.predict(std_data)
        st.write([[Pregnancies,Glucose,BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        diabetes_result=prediction2
        if(diabetes_result[0]==1):
            st.success("the Person has heart disease")
        else:
            st.success("the Person does not have heart disease")
       
        st.success(diabetes_result) 