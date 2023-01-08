# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 11:08:08 2022

@author: Shalom
"""

import numpy as np
import pickle
import streamlit as st
from PIL import Image 
image =Image.open('C:/Users\HP/Desktop/Streamlit App/image.JPG')


#loading the saved model
loaded_model= pickle.load(open('C:/Users\HP/Desktop/Streamlit App/trained_mode.sav', 'rb'))


#creating a functon for prediction 

def diabetes_prediction(input_data):
    
    #changing the input data to a numpy array
    input_data_as_numpy_arrary= np.asarray(input_data)

    #reshape the np as we predict for one instance
    input_data_reshape = input_data_as_numpy_arrary.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if(prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'the person is diabetic'
st.image(image)   

def main():
    
 #giving a title

 st.title('Diabetes Prediction Web App')
 
 #getting the input data from the user 
 

 
 
 Pregnancies = st.text_input('Number of Pregnancies')
 Glucose = st.text_input('Gluecose Level')
 BloodPressure= st.text_input('Your Blood Pressure')
 SkinThickness = st.text_input('Skin Thickness Value')
 Insulin= st.text_input('Insuline Level')
 BMI= st.text_input('BMI Value')
 DiabetesPedigreeFunction= st.text_input('DiabetesPedigreeFunction')
 Age= st.text_input('Your Age')
 
 
 #code for prediction
 
 diagnosis = ''
 
 
 #creating a button
 
 if st.button('Diabetes Test Results'):
     diagnosis=diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                    Insulin,BMI, DiabetesPedigreeFunction, Age])
    
 st.success(diagnosis)
 
 
if __name__ == '__main__':
    main()
st.text('BUILT BY SHALOM WITH 80% ACCURACY PERFORMANCE')