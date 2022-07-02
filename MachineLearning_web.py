from operator import mod
from statistics import mode
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error, r2_score;
from PIL import Image

#Asignamos un titulo
st.title('Machine Learning')

#Titulo de la side bar
st.sidebar.header('ALGORITMOS DE MACHINE LEARNING')

options = ['Regresión lineal','Regresión polinomial','Clasificador Gaussiano','Clasificador de árboles de desición','Redes neuronales']
model = st.sidebar.selectbox('Selecciona el algoritmo a utilizar',options)

st.subheader(model)

#Obetenemos el archivo
uploaded_file = st.file_uploader("Escoger archivo",type = ['csv','xls','xlsx','json'],help="¡Los archivos deben tener encabezados!")

global df,nombre_archivo
if uploaded_file is not None:

    #Obtenemos la extension del archivo que venga
    split_tup = os.path.splitext(uploaded_file.name)
    nombre_archivo = split_tup[0]
    extension = split_tup[1]

    if (extension == '.csv'):
        df = pd.read_csv(uploaded_file)
    elif(extension == '.xlsx'):
        df = pd.read_excel(uploaded_file)
      
try:
    st.write(df)
    headers = df.head()
    if(model == 'Regresión lineal'):
        st.write('PARAMETRIZACION')
        parameter_x = st.selectbox('Seleccione el parametro para el eje x',headers.columns)
        parameter_y = st.selectbox('Seleccione el parametro para el eje y',headers.columns)
        lin_reg = LinearRegression()
        #Preparar datos
        x_axis = df[parameter_x].values.reshape((-1,1)) #convertimos en un vector de vectores, en vez de un vector de datos
        y_axis = df[parameter_y]
        modelX = lin_reg.fit(x_axis,y_axis) #Ajustamos una linea a los datos que le daremos

        #Prediccion de tendencias
        y_pred = modelX.predict(x_axis)
        st.write('PREDICCION DE TENDENCIA Y FUNCION DE TENDENCIA LINEAL',y_pred)
        #Grafico de puntos de dispersion
        st.write('GRAFICO DE PUNTOS Y FUNCION DE TENDENCIA LINEAL')
        plt.scatter(x_axis,y_axis)
        plt.plot(x_axis,y_pred,color='red')
        plt.ylabel(parameter_x)
        plt.xlabel(parameter_y)
        plt.savefig('lin_reg.png')
        plt.close()
        image = Image.open('lin_reg.png')
        st.image(image, caption='Grafico de tendencia')
        st.write('CARACTERISTICAS')
        #Características
        st.write('Intersección (b) ', modelX.intercept_)
        st.write('Pendiente (m) ', modelX.coef_[0])
        st.write('Error medio: ', mean_squared_error(y_axis,y_pred,squared=True))
        st.write('R^2: ',r2_score(y_axis,y_pred))
        #Prediccion
        st.write('PREDICCION ESPECIFICA')
        number_input = st.number_input('Insertar número parapredicción de tendencia específica')
        st.write('Prediccion : ',modelX.predict([[number_input]])[0])

    elif(model == 'Regresión polinomial'):
        st.write('la del polinomio')
except Exception as e:
    print(e)
    st.write("Por favor subir archivo para poder ejecutar el algoritmo")





    
