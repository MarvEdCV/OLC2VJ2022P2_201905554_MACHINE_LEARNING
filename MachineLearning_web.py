import streamlit as st
import pandas as pd
import os
 
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

    st.write(extension)
    
    if (extension == '.csv'):
        df = pd.read_csv(uploaded_file)
    elif(extension == '.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    st.write("Nombre del archivo:",uploaded_file.name)
    
    
#try:
    st.write(df)
    if(model == 'Regresión lineal'):
        st.write('Ejecutar regresion lineal mijo')
    elif(model == 'Regresión polinomial'):
        st.write('la del polinomio')
#except Exception as e:
#    print(e)
#    st.write("Por favor subir archivo para poder ejecutar el algoritmo")





    
