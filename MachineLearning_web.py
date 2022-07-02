from operator import mod
from statistics import mode
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
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
        #Preparar datos
        x_axis = np.asarray(df[parameter_x]).reshape((-1,1)) #convertimos en un vector de vectores, en vez de un vector de datos
        y_axis = df[parameter_y]
        lin_reg = LinearRegression()
        lin_reg.fit(x_axis,y_axis) #Ajustamos una linea a los datos que le daremos

        #Prediccion de tendencias
        y_pred = lin_reg.predict(x_axis)
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
        st.write('Intersección (b) ', lin_reg.intercept_)
        st.write('Pendiente (m) ', lin_reg.coef_[0])
        st.write('Función de tendencia lineal(y=mx+b): y = ',lin_reg.coef_[0],'x + ',lin_reg.intercept_)
        st.write('Error medio: ', mean_squared_error(y_axis,y_pred,squared=True))
        st.write('R^2: ',r2_score(y_axis,y_pred))
        #Prediccion
        st.write('PREDICCION ESPECIFICA')
        number_input = st.number_input('Insertar número parapredicción de tendencia específica')
        st.write('Prediccion : ',lin_reg.predict([[number_input]])[0])
    elif(model == 'Regresión polinomial'):
        st.write('PARAMETRIZACION')
        parameter_x = st.selectbox('Seleccione el parametro para el eje x',headers.columns)
        parameter_y = st.selectbox('Seleccione el parametro para el eje y',headers.columns)
       
        #Preparar datos
        x_axis = np.asarray(df[parameter_x]).reshape((-1,1)) #convertimos en un vector de vectores, en vez de un vector de datos
        y_axis = df[parameter_y]
        #Grado del polinomio
        number_degree = st.number_input('Insertar número grado de polinomio',min_value=2)
        poly = PolynomialFeatures(degree=int(number_degree))
        x_trans = poly.fit_transform(x_axis)
      
        lin_reg = LinearRegression()
        lin_reg.fit(x_trans,y_axis) #Ajustamos una parabola a los datos que le daremos

        #Prediccion de tendencias
        y_pred = lin_reg.predict(x_trans)
        st.write('PREDICCION DE TENDENCIA Y FUNCION DE TENDENCIA POLINOMIAL',y_pred)
        #Grafico de puntos de dispersion
        st.write('GRAFICO DE PUNTOS Y FUNCION DE TENDENCIA LINEAL')
        plt.scatter(x_axis,y_axis)
        plt.plot(x_axis,y_pred,color='red')
        plt.ylabel(parameter_x)
        plt.xlabel(parameter_y)
        plt.savefig('pol_reg.png')
        plt.close()
        image = Image.open('pol_reg.png')
        st.image(image, caption='Grafico de tendencia')
        st.write('CARACTERISTICAS')
        #Características
        st.write('Error medio: ', np.sqrt(mean_squared_error(y_axis,y_pred,squared=True)))
        st.write('R^2: ',r2_score(y_axis,y_pred))
        cadena = ''
        size = len(lin_reg.coef_)
        for i in lin_reg.coef_:
            size = size-1
            cadena += '('+ str(i) + 'x^'+str(size)+')+'
        st.write('Función de tendencia polinomial(anx^n+an-1x^n-1+...+a0x^0): ',cadena[:-1])
        #Prediccion
        st.write('PREDICCION ESPECIFICA')
        number_input = st.number_input('Insertar número parapredicción de tendencia específica')
        x_new_min = int(number_input)
        x_new_max = int(number_input)
        x_new = np.linspace(x_new_min,x_new_max,1)
        x_new = x_new[:,np.newaxis]
        x_trans = poly.fit_transform(x_new)
        st.write('Prediccion : ',lin_reg.predict(x_trans))
except Exception as e:
    print(e)
    st.write("Por favor subir archivo para poder ejecutar el algoritmo")





    
