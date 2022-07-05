from operator import mod
from posixpath import split
from statistics import mode
from click import confirm
import streamlit as st
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import max_error, mean_squared_error, r2_score;
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
#Asignamos un titulo
st.title('Machine Learning')
st.subheader('Marvin Eduardo Catalán Véliz - 201905554')
#Titulo de la side bar
st.sidebar.header('ALGORITMOS DE MACHINE LEARNING')

options = ['Regresión lineal','Regresión polinomial','Clasificador Gaussiano','Clasificador de árboles de desición','Redes neuronales','Gauss sin label encoder','Arbol sin label encoder','Redes sin label encoder']
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
    elif(extension == '.xlsx' or extension == '.xls'):
        df = pd.read_excel(uploaded_file)
    elif(extension == '.json'):
        df = pd.read_json(uploaded_file)
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
    elif (model == 'Clasificador Gaussiano'):
        #number_input = st.number_input('Ingrese el numero de variables de entrada')
        entradas = st.multiselect('Seleccione las variables de entrada o variables independientes para el clasificador',headers.columns)
        salida = st.selectbox('Seleccione la variable de salida o variable dependiente',headers.columns)
        y = (df[salida].tolist())
        #Obtener las variables independientes
        feature = []
        size = len(entradas)
        le = preprocessing.LabelEncoder()
        for iterator in entradas:
            temporal = df[iterator].tolist()
            temporal2 = le.fit_transform(temporal)
            feature.append(temporal2)
            #Esto es sin label encoder
            #feature.append(df[iterator].tolist())
        st.write("PREDICCION")

        numbers = st.text_input("Ingrese los datos de predicción separados por coma en el orden que los selecciono arriba",help="Importante si el valor es True o False ponerlo en balor binario 1 o 0")
        x = list(zip(*feature))
        if(st.button('Ver arreglo clasificado')):
            st.write(x)
        if(st.button('Correr Clasificador Gaussiano')):
            clf = GaussianNB()
            #adaptacion de datos
            clf.fit(x,y)
            st.write('Predicción de X')
            st.write(clf.predict(x))
            if(len(numbers)>0):
                numeros_split = numbers.split(",")
                predict = [int(x) for x in numeros_split]
                predict = le.fit_transform(predict) #esto para label encoder, si no se quiere usar se borra
                try:
                    st.write('Predicción Especifica')
                    st.write(clf.predict([predict]))
                except Exception as e:
                    st.write(e)
                    st.write('Tiene datos incorrectos en el arreglo a predecir')
            else:
                st.write('El arreglo para la prediccion esta vacio o tiene datos incorrectos')
    elif(model == 'Clasificador de árboles de desición'):
        entradas = st.multiselect('Seleccione las variables de entrada o variables independientes para el clasificador',headers.columns)
        salida = st.selectbox('Seleccione la variable de salida o variable dependiente',headers.columns)
        y = (df[salida].tolist())
        #Obtener las variables independientes
        feature = []
        size = len(entradas)
        le = preprocessing.LabelEncoder()
        for iterator in entradas:
            temporal = df[iterator].tolist()
            temporal2 = le.fit_transform(temporal)
            feature.append(temporal2)
            #Esto es sin label encoder
            #feature.append(df[iterator].tolist())
        st.write("PREDICCION")
        x = list(zip(*feature))
        if(st.button('Ver arreglo clasificado')):
            st.write(x)
        if(st.button('Generar árbol de desición')):
            clf = DecisionTreeClassifier().fit(x,y)
            #adaptacion de datos
            plot_tree(clf, filled = True)
            plt.savefig('tree.png')
            plt.close()
            image = Image.open('tree.png')
            st.image(image, caption = 'Árbol de desición')
    elif (model == 'Gauss sin label encoder'):
        #number_input = st.number_input('Ingrese el numero de variables de entrada')
        entradas = st.multiselect('Seleccione las variables de entrada o variables independientes para el clasificador',headers.columns)
        salida = st.selectbox('Seleccione la variable de salida o variable dependiente',headers.columns)
        y = (df[salida].tolist())
        #Obtener las variables independientes
        feature = []
        size = len(entradas)
        le = preprocessing.LabelEncoder()
        for iterator in entradas:
            feature.append(df[iterator].tolist())
        st.write("PREDICCION")
        numbers = st.text_input("Ingrese los datos de predicción separados por coma en el orden que los selecciono arriba",help="Importante si el valor es True o False ponerlo en balor binario 1 o 0")
        x = list(zip(*feature))
        if(st.button('Ver arreglo clasificado')):
            st.write(x)
        if(st.button('Correr Clasificador Gaussiano')):
            clf = GaussianNB()
            #adaptacion de datos
            clf.fit(x,y)
            st.write('Predicción de X')
            st.write(clf.predict(x))
            if(len(numbers)>0):
                numeros_split = numbers.split(",")
                predict = [int(x) for x in numeros_split]
                #predict = le.fit_transform(predict) #esto para label encoder, si no se quiere usar se borra
                try:
                    st.write("Prediccion especifica")
                    st.write(clf.predict([predict]))
                except Exception as e:
                    st.write(e)
                    st.write('Tiene datos incorrectos en el arreglo a predecir')
            else:
                st.write('El arreglo para la prediccion esta vacio o tiene datos incorrectos')
    elif(model == 'Arbol sin label encoder'):
        entradas = st.multiselect('Seleccione las variables de entrada o variables independientes para el clasificador',headers.columns)
        salida = st.selectbox('Seleccione la variable de salida o variable dependiente',headers.columns)
        y = (df[salida].tolist())
        #Obtener las variables independientes
        feature = []
        size = len(entradas)
        le = preprocessing.LabelEncoder()
        for iterator in entradas:
            #temporal = df[iterator].tolist()
            #temporal2 = le.fit_transform(temporal)
            #feature.append(temporal2)
            #Esto es sin label encoder
            feature.append(df[iterator].tolist())
        st.write("PREDICCION")
        x = list(zip(*feature))
        if(st.button('Ver arreglo clasificado')):
            st.write(x)
        if(st.button('Generar árbol de desición')):
            clf = DecisionTreeClassifier().fit(x,y)
            #adaptacion de datos
            plot_tree(clf, filled = True)
            plt.savefig('tree.png')
            plt.close()
            image = Image.open('tree.png')
            st.image(image, caption = 'Árbol de desición')
    elif(model == 'Redes neuronales'):
        entradas = st.multiselect('Seleccione las variables de entrada o variables independientes para el clasificador',headers.columns)
        salida = st.selectbox('Seleccione la variable de salida o variable dependiente',headers.columns)
        y = (df[salida].tolist())
        #Obtener las variables independientes
        feature = []
        le = preprocessing.LabelEncoder()
        for iterator in entradas:
            temporal = df[iterator].tolist()
            temporal2 = le.fit_transform(temporal)
            feature.append(temporal2.tolist())
            #Esto es sin label encoder
            #feature.append(df[iterator].tolist())
        st.write("PREDICCION")
        x = list(zip(*feature))
        numbers = st.text_input("Ingrese los datos de predicción separados por coma en el orden que los selecciono arriba",help="Importante si el valor es True o False ponerlo en balor binario 1 o 0")
        if(st.button('Ver arreglo clasificado')):
            st.write(x)
        if(st.button('Generar redes neuronales')):
            scaler = StandardScaler()
            scaler.fit(feature,y)            
            mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000,solver='lbfgs')
            mlp.fit(x,y)

            #PREDICCIONES PARA X 
            st.write('TABLA DE PREDICCIONES PARA X')
            pred = mlp.predict(x)
            st.write(pred)

            #PREDICCIONES PARA 
            if(len(numbers)>0):
                numeros_split = numbers.split(",")
                predict = [int(x) for x in numeros_split]
                predict = le.fit_transform(predict) #esto para label encoder, si no se quiere usar se borra
                try:
                    st.write('PREDICCION ESPEFICIFA')
                    st.write(mlp.predict([predict]))
                except Exception as e:
                    st.write(e)
                    st.write('Tiene datos incorrectos en el arreglo a predecir')
            else:
                st.write('El arreglo para la prediccion esta vacio o tiene datos incorrectos') 
    elif(model == 'Redes sin label encoder'):
        entradas = st.multiselect('Seleccione las variables de entrada o variables independientes para el clasificador',headers.columns)
        salida = st.selectbox('Seleccione la variable de salida o variable dependiente',headers.columns)
        y = (df[salida].tolist())
        #Obtener las variables independientes
        feature = []
        le = preprocessing.LabelEncoder()
        for iterator in entradas:
            #temporal = df[iterator].tolist()
            #temporal2 = le.fit_transform(temporal)
            #feature.append(temporal2.tolist())
            #Esto es sin label encoder
            feature.append(df[iterator].tolist())
        st.write("PREDICCION")
        x = list(zip(*feature))
        numbers = st.text_input("Ingrese los datos de predicción separados por coma en el orden que los selecciono arriba",help="Importante si el valor es True o False ponerlo en balor binario 1 o 0")
        if(st.button('Ver arreglo clasificado')):
            st.write(x)
        if(st.button('Generar redes neuronales')):
            #lista = []
            #lista.append(feature)
            #print(feature)
            #print(df[['A','B','C','D']])
            #print(y)
            #X_train , X_test, y_train, y_test = train_test_split(feature,y)
            scaler = StandardScaler()
            scaler.fit(feature,y)
            #X_train = scaler.transform(feature)
            #X_test = scaler.transform(X_test)
            
            mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000,solver='lbfgs')
            mlp.fit(x,y)

            #PREDICCIONES PARA X 
            st.write('TABLA DE PREDICCIONES PARA X')
            pred = mlp.predict(x)
            st.write(pred)

            #PREDICCIONES PARA 
            if(len(numbers)>0):
                numeros_split = numbers.split(",")
                predict = [int(x) for x in numeros_split]
                #predict = le.fit_transform(predict) #esto para label encoder, si no se quiere usar se borra
                try:
                    st.write('PREDICCION ESPEFICIFA')
                    st.write(mlp.predict([predict]))
                except Exception as e:
                    st.write(e)
                    st.write('Tiene datos incorrectos en el arreglo a predecir')
            else:
                st.write('El arreglo para la prediccion esta vacio o tiene datos incorrectos') 
except Exception as e:
    print(e)
    st.write("Por favor subir archivo para poder ejecutar el algoritmo")