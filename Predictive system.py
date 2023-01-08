# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle


#loading the saved model
loaded_model= pickle.load(open('C:/Users\HP/Desktop/Streamlit App/trained_mode.sav', 'rb'))


input_data =(3,126,88,41,235,39.3,0.704,27)

#changing the input data to a numpy array
input_data_as_numpy_arrary= np.asarray(input_data)

#reshape the np as we predict for one instance
input_data_reshape = input_data_as_numpy_arrary.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if(prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('the person is diabetic')
