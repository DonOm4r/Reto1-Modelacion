import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.sidebar.title("Prediccion con Modelos de Regresión")
# Sidebar para la carga del archivo CSV
uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")

# Inicialización de variables
Misdatos = None
predictions = []
RMSE_array = []

def Cal_error(Y_prediction,Y):
  MSE=np.mean(np.square(Y_prediction - Y))
  RMSE=np.sqrt(MSE)
  return (MSE,RMSE)

# Verifica si se ha subido un archivo
if uploaded_file is not None:
    Misdatos = pd.read_csv(uploaded_file)
    # Procesamiento de los datos...
    # Asegúrate de calcular tus predicciones aquí y llenar la lista 'predictions'
    # ...

    Misdatos_Array = np.array(Misdatos)

    X = Misdatos_Array[:,0]
    Y = Misdatos_Array[:,1]
    #Minimos cuadrados
    X2 = X**2
    X3 = X**3
    X4 = X**4

    XY = X*Y
    X2Y = X2*Y

    #Generamos las sumatorias de los vectores
    SumaX = sum(X)
    SumaX2 = sum(X2)
    SumaX3 = sum(X3)
    SumaX4 = sum(X4)

    SumaY = sum(Y)
    SumaXY = sum(XY)
    SumaX2Y = sum(X2Y)

    #Armamos la matriz y el vector
    Matriz = np.array([[len(X),SumaX],[SumaX,SumaX2]])
    Vector = np.array([[SumaY],[SumaXY]])

    #Solucionamos el sistema
    SL = np.dot(np.linalg.inv(Matriz),Vector)

    #Armando el modelo
    K1 = SL[0]
    K2 = SL[1]

    Y_prediction = K1 + K2*X

    predictions.append((Y_prediction, "ORDEN 1"))

    #Armamos la matriz y el vector
    Matriz = np.array([[len(X),SumaX,SumaX2],[SumaX,SumaX2,SumaX3],[SumaX2,SumaX3,SumaX4]])
    Vector = np.array([[SumaY],[SumaXY],[SumaX2Y]])

    #Solucionamos el sistema
    SL = np.dot(np.linalg.inv(Matriz),Vector)

    #Armando el modelo
    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]

    Y_prediction = K1 + K2*X + K3*X2

    predictions.append((Y_prediction, "ORDEN 2"))

    #ORDEN 3
    X2 = X**2
    X3 = X**3
    X4 = X**4
    X5 = X**5
    X6 = X**6

    XY = X*Y
    X2Y = X2*Y
    X3Y = X3*Y

    #Generamos las sumatorias de los vectores
    SumaX = sum(X)
    SumaX2 = sum(X2)
    SumaX3 = sum(X3)
    SumaX4 = sum(X4)
    SumaX5 = sum(X5)
    SumaX6 = sum(X6)

    SumaY = sum(Y)
    SumaXY = sum(XY)
    SumaX2Y = sum(X2Y)
    SumaX3Y = sum(X3Y)

    #Armamos la matriz y el vector
    Matriz = np.array([[len(X),SumaX,SumaX2,SumaX3],[SumaX,SumaX2,SumaX3,SumaX4],[SumaX2,SumaX3,SumaX4,SumaX5],[SumaX3,SumaX4,SumaX5,SumaX6]])
    Vector = np.array([[SumaY],[SumaXY],[SumaX2Y],[SumaX3Y]])

    #Solucionamos el sistema
    SL = np.dot(np.linalg.inv(Matriz),Vector)

    #Armando el modelo
    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3

    predictions.append((Y_prediction, "ORDEN 3"))

    # Orden 4
    X7 = X**7
    X8 = X**8
    X4Y = X4*Y

    SumaX7 = sum(X7)
    SumaX8 = sum(X8)
    SumaX4Y = sum(X4Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8]
                      ])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4

    predictions.append((Y_prediction, "ORDEN 4"))

    # Orden 5
    X9 = X**9
    X10 = X**10

    X5Y = X5 * Y

    SumaX8 = sum(X8)
    SumaX9 = sum(X9)
    SumaX10 = sum(X10)
    SumaX5Y = sum(X5Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10]
                      ])

    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5

    predictions.append((Y_prediction, "ORDEN 5"))


    #orden 6
    X11 = X ** 11
    X12 = X ** 12

    X6Y = X**6 * Y

    SumaX11 = sum(X11)
    SumaX12 = sum(X12)
    SumaX6Y = sum(X6Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6

    predictions.append((Y_prediction, "ORDEN 6"))

    # Orden 7
    X13 = X**13
    X14 = X**14
    X7Y = X**7 * Y

    SumaX13 = sum(X13)
    SumaX14 = sum(X14)
    SumaX7Y = sum(X7Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7

    predictions.append((Y_prediction, "ORDEN 7"))

    # Orden 8
    X15 = X**15
    X16 = X**16
    X8Y = X**8 * Y

    SumaX15 = sum(X15)
    SumaX16 = sum(X16)
    SumaX8Y = sum(X8Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y],[SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8

    predictions.append((Y_prediction, "ORDEN 8"))

    # Orden 9
    X17 = X**17
    X18 = X**18
    X9Y = X**9 * Y

    SumaX17 = sum(X17)
    SumaX18 = sum(X18)
    SumaX9Y = sum(X9Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9

    predictions.append((Y_prediction, "ORDEN 9"))

    # Orden 10
    X19 = X**19
    X20 = X**20
    X10Y = X**10 * Y

    SumaX19 = sum(X19)
    SumaX20 = sum(X20)
    SumaX10Y = sum(X10Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10

    predictions.append((Y_prediction, "ORDEN 10"))


    # Orden 11
    X21 = X**21
    X22 = X**22
    X11Y = X**11 * Y

    SumaX21 = sum(X21)
    SumaX22 = sum(X22)
    SumaX11Y = sum(X11Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11

    predictions.append((Y_prediction, "ORDEN 11"))

    # Orden 12
    X23 = X**23
    X24 = X**24
    X12Y = X**12 * Y

    SumaX23 = sum(X23)
    SumaX24 = sum(X24)
    SumaX12Y = sum(X12Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12

    predictions.append((Y_prediction, "ORDEN 12"))


    # Orden 13
    X25 = X**25
    X26 = X**26
    X13Y = X**13 * Y

    SumaX25 = sum(X25)
    SumaX26 = sum(X26)
    SumaX13Y = sum(X13Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13

    predictions.append((Y_prediction, "ORDEN 13"))

    # Orden 14
    X27 = X**27
    X28 = X**28
    X14Y = X**14 * Y

    SumaX27 = sum(X27)
    SumaX28 = sum(X28)
    SumaX14Y = sum(X14Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27],
                      [SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y], [SumaX14Y]])
    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]
    K15 = SL[14]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13 + K15*X14

    predictions.append((Y_prediction, "ORDEN 14"))

    # Orden 15
    X29 = X**29
    X30 = X**30
    X15Y = X**15 * Y

    SumaX29 = sum(X29)
    SumaX30 = sum(X30)
    SumaX15Y = sum(X15Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28],
                      [SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29],
                      [SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y], [SumaX14Y], [SumaX15Y]])
    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]
    K15 = SL[14]
    K16 = SL[15]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13 + K15*X14 + K16*X15

    predictions.append((Y_prediction, "ORDEN 15"))

    # Orden 16
    X31 = X**31
    X32 = X**32
    X16Y = X**16 * Y

    SumaX31 = sum(X31)
    SumaX32 = sum(X32)
    SumaX16Y = sum(X16Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29],
                      [SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30],
                      [SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31],
                      [SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32]])
    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y], [SumaX14Y], [SumaX15Y],[SumaX16Y]])
    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]
    K15 = SL[14]
    K16 = SL[15]
    K17 = SL[16]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13 + K15*X14 + K16*X15 + K17*X16

    predictions.append((Y_prediction, "ORDEN 16"))

    # Orden 17
    X33 = X**33
    X34 = X**34
    X17Y = X**17 * Y

    SumaX33 = sum(X33)
    SumaX34 = sum(X34)
    SumaX17Y = sum(X17Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30],
                      [SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31],
                      [SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32],
                      [SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33],
                      [SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34]])

    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y], [SumaX14Y], [SumaX15Y], [SumaX16Y],[SumaX17Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]
    K15 = SL[14]
    K16 = SL[15]
    K17 = SL[16]
    K18 = SL[17]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13 + K15*X14 + K16*X15 + K17*X16 + K18*X17

    predictions.append((Y_prediction, "ORDEN 17"))


    # Orden 18
    X35 = X**35
    X36 = X**36
    X18Y = X**18 * Y

    SumaX35 = sum(X35)
    SumaX36 = sum(X36)
    SumaX18Y = sum(X18Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31],
                      [SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32],
                      [SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33],
                      [SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34],
                      [SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35],
                      [SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36]])

    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y], [SumaX14Y], [SumaX15Y], [SumaX16Y], [SumaX17Y], [SumaX18Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]
    K15 = SL[14]
    K16 = SL[15]
    K17 = SL[16]
    K18 = SL[17]
    K19 = SL[18]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13 + K15*X14 + K16*X15 + K17*X16 + K18*X17 + K19*K18

    predictions.append((Y_prediction, "ORDEN 18"))

    # Orden 19
    X37 = X**37
    X38 = X**38
    X19Y = X**19 * Y

    SumaX37 = sum(X37)
    SumaX38 = sum(X38)
    SumaX19Y = sum(X19Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32],
                      [SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33],
                      [SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34],
                      [SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35],
                      [SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36],
                      [SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36, SumaX37],
                      [SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36, SumaX37, SumaX38]])

    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y], [SumaX14Y], [SumaX15Y], [SumaX16Y], [SumaX17Y], [SumaX18Y], [SumaX19Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]
    K15 = SL[14]
    K16 = SL[15]
    K17 = SL[16]
    K18 = SL[17]
    K19 = SL[18]
    K20 = SL[19]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13 + K15*X14 + K16*X15 + K17*X16 + K18*X17 + K19*X18 + K20*K19

    predictions.append((Y_prediction, "ORDEN 19"))

    # Orden 20
    X39 = X**39
    X40 = X**40
    X20Y = X**20 * Y

    SumaX39 = sum(X39)
    SumaX40 = sum(X40)
    SumaX20Y = sum(X20Y)

    Matriz = np.array([[len(X), SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20],
                      [SumaX, SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21],
                      [SumaX2, SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22],
                      [SumaX3, SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23],
                      [SumaX4, SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24],
                      [SumaX5, SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25],
                      [SumaX6, SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26],
                      [SumaX7, SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27],
                      [SumaX8, SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28],
                      [SumaX9, SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29],
                      [SumaX10, SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30],
                      [SumaX11, SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31],
                      [SumaX12, SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32],
                      [SumaX13, SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33],
                      [SumaX14, SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34],
                      [SumaX15, SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35],
                      [SumaX16, SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36],
                      [SumaX17, SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36, SumaX37],
                      [SumaX18, SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36, SumaX37, SumaX38],
                      [SumaX19, SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36, SumaX37, SumaX38, SumaX39],
                      [SumaX20, SumaX21, SumaX22, SumaX23, SumaX24, SumaX25, SumaX26, SumaX27, SumaX28, SumaX29, SumaX30, SumaX31, SumaX32, SumaX33, SumaX34, SumaX35, SumaX36, SumaX37, SumaX38, SumaX39, SumaX40]])

    Vector = np.array([[SumaY], [SumaXY], [SumaX2Y], [SumaX3Y], [SumaX4Y], [SumaX5Y], [SumaX6Y], [SumaX7Y], [SumaX8Y], [SumaX9Y], [SumaX10Y], [SumaX11Y], [SumaX12Y], [SumaX13Y], [SumaX14Y], [SumaX15Y], [SumaX16Y], [SumaX17Y], [SumaX18Y], [SumaX19Y], [SumaX20Y]])

    SL = np.dot(np.linalg.inv(Matriz), Vector)

    K1 = SL[0]
    K2 = SL[1]
    K3 = SL[2]
    K4 = SL[3]
    K5 = SL[4]
    K6 = SL[5]
    K7 = SL[6]
    K8 = SL[7]
    K9 = SL[8]
    K10 = SL[9]
    K11 = SL[10]
    K12 = SL[11]
    K13 = SL[12]
    K14 = SL[13]
    K15 = SL[14]
    K16 = SL[15]
    K17 = SL[16]
    K18 = SL[17]
    K19 = SL[18]
    K20 = SL[19]
    K21 = SL[20]

    Y_prediction = K1 + K2*X + K3*X2 + K4*X3 + K5*X4 + K6*X5 + K7*X6 + K8*X7 + K9*X8 + K10*X9 + K11*X10 + K12*X11 + K13*X12 + K14*X13 + K15*X14 + K16*X15 + K17*X16 + K18*X17 + K19*X18 + K20*X19 + K21*X20

    predictions.append((Y_prediction, "ORDEN 20"))

    # Sidebar para visualizar la tabla completa
    if st.sidebar.checkbox('Ver tabla completa'):
        # Crear dos columnas
        col1, col2 = st.columns(2)
    
        # Columna para la tabla
        with col1:
          st.title("Tabla")
          st.write(Misdatos)
    
        # Columna para la gráfica
        with col2:
          st.title("Gráfica")
          fig1, ax1 = plt.subplots()
          ax1.plot(X, Y, '*r', label='Datos')
          plt.xlabel('Tiempo')
          plt.ylabel('Temperatura')
          plt.title('Ebullicion de el Agua')
          plt.grid()
          st.pyplot(fig1)

    # Sidebar para ingresar el orden deseado
    orden_deseado = st.sidebar.number_input('Ingresa el orden deseado', min_value=1, max_value=20, value=1, step=1)
    st.title("Orden deseado")
    # Función para mostrar la gráfica y el error del orden deseado
    def mostrar_grafica_y_error(orden_deseado, calcular_optimo=False):
        # Verifica que las predicciones estén disponibles
        if not predictions:
            st.error("No hay predicciones disponibles. Asegúrate de haber cargado los datos y calculado las predicciones.")
            return
        
        # Encuentra la predicción correspondiente al orden deseado
        for Y_pred, title in predictions:
            if f"ORDEN {orden_deseado}" in title:
                fig, ax = plt.subplots()
                ax.scatter(X, Y, label='Datos')
                ax.plot(X, Y_pred, '*k', label='Predicción Modelo')
                ax.set_title(f"Modelo de Regresión - {title}")
                ax.set_xlabel("Tiempo")
                ax.set_ylabel("Temperatura")
                ax.legend()
                st.pyplot(fig)

                # Calcula y muestra el error
                [MSE, RMSE] = Cal_error(Y_pred, Y)
                st.write(f'El valor de error RMSE para {title} es: ', RMSE)

                if calcular_optimo:
                  RMSE_array.append(RMSE)
                break

    # Si se ingresa un orden, mostrar solo la gráfica de ese orden
    if orden_deseado:
        mostrar_grafica_y_error(orden_deseado)

    # Sidebar para mostrar todos los órdenes
    if st.sidebar.checkbox('Mostrar todos los órdenes'):
        RMSE_array.clear()
        st.title("Todos los modelos")
        for i in range(1, 21):
            mostrar_grafica_y_error(i, calcular_optimo=True)
        orden_optimo = np.argmin(RMSE_array) + 1
        st.sidebar.success(
          f'El orden óptimo recomendado es: {orden_optimo} ya que su valor de error RMSE es de: '
          f'{RMSE_array[orden_optimo-1]} siendo este el más bajo de los modelos calculados.')