# -*- coding: utf-8 -*-
"""
UNIVERSIDAD NACONAL AUTONOMA DE MEXICO
FACULTAD DE INGENIERIA
******************************
******************************
Autor: González Ochoa José Antonio
Entrega final proyecto IA

Se implementan algunos algoritmos de IA:
*Reglas de asociación
*Matriz de correlaciones
*Métricas de similitud.
*Clustering particional
*Regresion logistica
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from apyori import apriori 

from tkinter import *
from tkinter import ttk
from tkinter import Button
from tkinter import Frame
from tkinter import Tk
from tkinter import filedialog
from tkinter import Label
from tkinter import Radiobutton
from tkinter import Checkbutton
from tkinter import Spinbox
from tkinter import OptionMenu
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


global Clasificacion
BreastCancer = pd.read_csv('D:\SEPTIMO SEMESTRE\Inteligencia artificial\PROYECTO\WDBCOriginal.csv')
BCancer = BreastCancer.replace({'M': 0, 'B': 1})    
X = np.array(BCancer[['Texture', 'Area', 'Compactness','Concavity', 'Symmetry', 'FractalDimension']])
Y = np.array(BCancer[['Diagnosis']])
Clasificacion = linear_model.LogisticRegression()
validation_size = 0.2
seed = 1234
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed, shuffle = True)
Clasificacion.fit(X_train, Y_train)
Probabilidad = Clasificacion.predict_proba(X_train)
Predicciones = Clasificacion.predict(X_train)
PrediccionesNuevas = Clasificacion.predict(X_validation)
confusion_matrix = pd.crosstab(Y_validation.ravel(), PrediccionesNuevas, rownames=['Real'], colnames=['Predicción'])


"""
Se define la capa base
"""


raiz = Tk()
raiz.title("Algoritmos IA")
miFrame = Frame(raiz)
miFrame.pack()
botonHerramientas = Button(miFrame, text = "Herramientas")
botonHerramientas.grid(row = 0, column = 0, padx=5,pady=5)

"""Funcion encargada de abrir el archivo y obtener la ruta del mismo"""
def abrirArchivo():
    global rutaArchivo
    nombreRutaArchivo = filedialog.askopenfilename(initialdir = "/", title = "Seleccionar Archivo",
                                                filetypes = (("txt files","*.txt"), ("csv files","*.csv"),
                                                ("xls files","*.xls")))
    rutaArchivo = nombreRutaArchivo
    
"""Funcion encargada de obtener el df del archivo y realizar la lectura segun 
el tipo de archivo y formato"""
def obtenerDatosArchivo():
    
    global datosArchivo
   
    if entradaSeparador.get() == "," and estadoEncabezado.get() == 1 and estadoIndexCol.get() == 1:
        datosArchivo = pd.read_csv(rutaArchivo, index_col = 0)
    elif entradaSeparador.get() == "," and estadoEncabezado.get() == 0 and estadoIndexCol.get() == 0:
        datosArchivo = pd.read_csv(rutaArchivo, header = None)
    elif entradaSeparador.get() == "," and estadoEncabezado.get() == 0 and estadoIndexCol.get() == 1:
        datosArchivo = pd.read_csv(rutaArchivo, header = None, index_col= 0)
    elif entradaSeparador.get() == "," and estadoEncabezado.get() == 1 and estadoIndexCol.get() == 0:
        datosArchivo = pd.read_csv(rutaArchivo)
    elif entradaSeparador.get() == "" and estadoEncabezado.get() == 1 and estadoIndexCol.get() == 1:
        datosArchivo = pd.read_table(rutaArchivo, index_col = 0)
    elif entradaSeparador.get() == "" and estadoEncabezado.get() == 0 and estadoIndexCol.get() == 0:
        datosArchivo = pd.read_table(rutaArchivo, header = None)
    elif entradaSeparador.get() == "" and estadoEncabezado.get() == 0 and estadoIndexCol.get() == 1:
        datosArchivo = pd.read_table(rutaArchivo, header = None, index_col = 0)
    elif entradaSeparador.get() == "" and estadoEncabezado.get() == 1 and estadoIndexCol.get() == 0:
        datosArchivo = pd.read_table(rutaArchivo)
    """print(datosArchivo) """   
    
"""Funcion encargada de obtener la reglas de asociacion"""       
def obtenerReglasAsociacion():
    global registros
    global reglas
    global resultados
    global totalReglas
    registros = []
    for i in range(0, len(datosArchivo.axes[0])):
        registros.append([str(datosArchivo.values[i,j]) for j in range(0, len(datosArchivo.axes[1]))])
    reglas = apriori(registros, min_support = float(rangoApoyo.get()), min_confidence = float(rangoConfianza.get()), 
                     min_lift = float(rangoMinLift.get()), min_length =float( rangoMinL.get()))
    resultados = list(reglas)
    totalReglas = len(resultados)
    """print(totalReglas)
    print(resultados)"""
    
"""Funcion encargada de crear un data frame con base a las reglas obtenidas,
dando un formato más legible y manipulable"""   
def creaDFReglasOrdenadas():
    global df
    df = pd.DataFrame(columns=('Items','Antecedent','Consequent','Support','Confidence','Lift'))
    global Support
    global Confidence
    global Lift
    Support =[]
    Confidence = []
    Lift = []
    Items = []
    Antecedent = []
    Consequent=[]
    
    for RelationRecord in resultados:
        for ordered_stat in RelationRecord.ordered_statistics:
            if Support == [] or RelationRecord.items != Items[-1]:
                Support.append(RelationRecord.support)
                Items.append(RelationRecord.items)
                Antecedent.append(ordered_stat.items_base)
                Consequent.append(ordered_stat.items_add)
                Confidence.append(ordered_stat.confidence)
                Lift.append(ordered_stat.lift)
            elif RelationRecord.items == Items[-1] and  RelationRecord.support != Support[-1] and ordered_stat.confidence != Confidence[-1] and ordered_stat.lift != Lift[-1] :
                Support.append(RelationRecord.support)
                Items.append(RelationRecord.items)
                Antecedent.append(ordered_stat.items_base)
                Consequent.append(ordered_stat.items_add)
                Confidence.append(ordered_stat.confidence)
                Lift.append(ordered_stat.lift)
            
    
    df['Items'] = list(map(set, Items))                                   
    df['Antecedent'] = list(map(set, Antecedent))
    df['Consequent'] = list(map(set, Consequent))
    df['Support'] = Support
    df['Confidence'] = Confidence
    df['Lift']= Lift
    
    df.sort_values(by =opcionSort.get(), ascending = False, inplace = True)
    cajaTextoReglas.insert(INSERT, str(df) + "\n" )
    #print(df)
    #print(len(df))
 

"""Funcion encargada de mostrar las reglas de asociacion"""       
def mostrarReglasAsociacion():
    obtenerReglasAsociacion()
    creaDFReglasOrdenadas()
    cajaTextoReglas.insert(INSERT, "Reglas generadas: " + str(totalReglas) + "\n")
    cajaTextoReglas.insert(INSERT, "=====================================\n")

"""Funcion encargada de graficar las reglas de asociacion"""
def graficarRA():
    liftSort = sorted(Lift)
    graficaRA = plt.scatter(Support, Confidence, c = liftSort,alpha = 0.3)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    cbar1 = plt.colorbar(graficaRA)
    cbar1.set_label("Lift")
    plt.show()

def obtenerMatrizCor():
    global matrizCor
    metodo = opcionMetod.get()
    matrizCor = datosArchivo.corr(method = metodo.lower())

def mostrarMatrizCor():
    obtenerMatrizCor()
    #print(matrizCor)
    cajaTextoMatrizCor.insert(INSERT, matrizCor)
    
def graficarCor():
    plt.matshow(matrizCor)
    plt.xticks(range(len(matrizCor.columns)), matrizCor.columns, rotation = 90)
    plt.yticks(range(len(matrizCor.columns)), matrizCor.columns)
    plt.colorbar()
    plt.show()
    
"""Funcion encargada de construir la matriz de distancias segun el metodo elegido"""   
def construyeMatrizDistancias():
	numeroRegistros = len(datosArchivo.axes[0])
	metodoDistancia = opcionMetodDist.get()
	matrizDF = pd.DataFrame()
	distancias = []
	if metodoDistancia == "Euclidean":
		for i in range(0, numeroRegistros):
			for j in range(0, numeroRegistros):
				ri = datosArchivo.iloc[i].values
				rj = datosArchivo.iloc[j].values
				dst = distance.euclidean(ri,rj)
				distancias.append(dst)
			matrizDF[i] = distancias
			distancias[:] = []

	elif metodoDistancia == "Chebyshev":
		for i in range(0, numeroRegistros):
			for j in range(0, numeroRegistros):
				ri = datosArchivo.iloc[i].values
				rj = datosArchivo.iloc[j].values
				dst = distance.chebyshev(ri,rj)
				distancias.append(dst)
			matrizDF[i] = distancias
			distancias[:] = []

	elif metodoDistancia == "Cityblock":
		for i in range(0, numeroRegistros):
			for j in range(0, numeroRegistros):
				ri = datosArchivo.iloc[i].values
				rj = datosArchivo.iloc[j].values
				dst = distance.cityblock(ri,rj)
				distancias.append(dst)
			matrizDF[i] = distancias
			distancias[:] = []

	elif metodoDistancia == "Minkowski":
		for i in range(0, numeroRegistros):
			for j in range(0, numeroRegistros):
				ri = datosArchivo.iloc[i].values
				rj = datosArchivo.iloc[j].values
				dst = distance.minkowski(ri,rj)
				distancias.append(dst)
			matrizDF[i] = distancias
			distancias[:] = []
	cajaTextoDistancias.insert(INSERT, matrizDF)

def variablesSeleccionadas():
    variables = list(entradaColumnas.get().split(","))
    variablesInt = list(map(int, variables)) 

    global variablesModelo 
    variablesModelo = datosArchivo.iloc[:, variablesInt].values
    global SSE 
    SSE = []

    for i in range(2, 16):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(variablesModelo)
        SSE.append(km.inertia_)

def grafica_elbowCurve():
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 16), SSE, marker='o')
    plt.xlabel('Cantidad de clusters *k*')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()

def kMeans():
    global CentroidesP
    global MParticional
    MParticional = KMeans(n_clusters=int(rangoCluster.get()), random_state=0).fit(variablesModelo)
    MParticional.predict(variablesModelo)
    CentroidesP = MParticional.cluster_centers_
    cajaTextoCluster.insert(INSERT, "Se obtuvieron los siguientes centroides: \n")
    cajaTextoCluster.insert(INSERT, CentroidesP)


def graficaCentros():
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.style.use('ggplot')
    colores=['red', 'blue', 'cyan', 'green', 'yellow']
    asignar=[]
    for row in MParticional.labels_:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter (variablesModelo[:, 0], variablesModelo[:, 1], variablesModelo[:, 2], marker='o', c=asignar, s=60)
    ax.scatter(CentroidesP[:, 0], CentroidesP[:, 1], CentroidesP[:, 2], marker='*', c=colores, s=1000)
    plt.show()


def diagnosticaPaciente():
    Paciente = pd.DataFrame({'Texture': [entradaTexture.get()], 'Area': [entradaArea.get()], 'Compactness': [entradaCompactness.get()],'Concavity': [entradaConcavity.get()], 'Symmetry': [entradaSymmetry.get()], 'FractalDimension': [entradaFD.get()]})
    diagnostico = Clasificacion.predict(Paciente)
    cajaTextoCancer.insert(INSERT, "\nEl paciente con ID " + entradaID.get() +" presenta el siguiente diagnostico:\n")
    if diagnostico == [0]:
        cajaTextoCancer.insert(INSERT, "    padece de un tumor malingo\n")
    else:
        cajaTextoCancer.insert(INSERT, "    padece de un tumor benigno\n")




botonEjecutar= Button(miFrame, text = "Ejecutar", command = obtenerDatosArchivo)    
botonEjecutar.grid(row = 1, column = 0, padx=5,pady=5)
botonNuevo = Button(miFrame, text = "Nuevo")
botonNuevo.grid(row = 1, column = 1, padx=5,pady=5)
botonAbrir = Button(miFrame, text = "Abrir")
botonAbrir.grid(row = 1, column = 2, padx=5,pady=5)
botonGuardar = Button(miFrame, text = "Guardar")
botonGuardar.grid(row = 1, column = 3, padx=5,pady=5)

"""
Se definen las pestañas
"""
pestañas = ttk.Notebook(raiz)
pestañas.pack(fill = 'both', expand = 'yes')
p1 = ttk.Frame(pestañas)
p2 = ttk.Frame(pestañas)
p5 = ttk.Frame(pestañas)
p6 = ttk.Frame(pestañas)
p9 = ttk.Frame(pestañas)
p10 = ttk.Frame(pestañas)
#pestañas.bind("<<NotebookTabChanged>>", on_tab_selected) #avisa si hay un evento de pestaña
pestañas.add(p1, text = "Datos")
pestañas.add(p2, text = "Correlación")
pestañas.add(p5, text = "Clúster")
pestañas.add(p6, text = "Asociada")
pestañas.add(p9, text = "Similitudes")
pestañas.add(p10, text = "Análsis Cáncer")

"""
Se define el botno archivo junto con su funcionamiento
"""

textoArchivo = Label(p1, text = "Archivo:")
textoArchivo.grid(row=4, column=0, padx=5,pady=5)

    
abreArchivo = Button(p1, text = "Buscar Archivo", command = abrirArchivo)
abreArchivo.grid(row = 4, column = 1, padx =5, pady = 5)
textoSeparador = Label(p1, text = "Separador")
textoSeparador.grid(row = 4, column = 2, padx =5, pady = 5)
entradaSeparador =  Entry(p1, width=4)
entradaSeparador.grid(row = 4, column = 3, padx =5, pady = 5)
textoDecimal = Label(p1, text = "Decimal")
textoDecimal.grid(row = 4, column = 4, padx =5, pady = 5)
entradaDecimal =  Entry(p1, width=4)
entradaDecimal.grid(row = 4, column = 5, padx =5, pady = 5)
estadoEncabezado = IntVar()
checkEncabezado= Checkbutton(p1, text = "Encabezado", variable = estadoEncabezado)
checkEncabezado.grid(row=4, column=6, padx=5,pady=5)
estadoIndexCol = IntVar()
checkIndexCol= Checkbutton(p1, text = "Columna Indice", variable = estadoIndexCol)
checkIndexCol.grid(row=4, column=7, padx=5,pady=5)
"""
Se definen los botones de la página 2
"""
frameBotonesP2 = Frame(p2)
frameBotonesP2.grid(row = 4, column = 0)        
"""Botones Correlacion"""
textoMetodo = Label(frameBotonesP2, text = "Método:")
textoMetodo.grid(row=5, column=0, padx=5,pady=5)

opcionMetod = StringVar(frameBotonesP2)
opcionMetod.set("Pearson")
seleccionMetod = OptionMenu(frameBotonesP2, opcionMetod, "Pearson", "Kendall","Spearman")
seleccionMetod.grid(row=5, column=1, padx=5,pady=5)

botonMatrizCor= Button(frameBotonesP2, text = "Matriz Correlaciones", command = mostrarMatrizCor)    
botonMatrizCor.grid(row = 5, column = 2, padx=5,pady=5)

botonGraficaCor= Button(frameBotonesP2, text = "Grafica Correlaciones", command = graficarCor)    
botonGraficaCor.grid(row = 5, column = 3, padx=5,pady=5)

frameMatrizCor = Frame(p2)
frameMatrizCor.grid(row = 6, column = 0)
cajaTextoMatrizCor = Text(frameMatrizCor, width = 150, height = 20)
cajaTextoMatrizCor.grid(row = 7, column = 1, padx = 5, pady = 5)
scrollMatrizCor = Scrollbar(frameMatrizCor, command = cajaTextoMatrizCor.yview)
scrollMatrizCor.grid(row = 7, column = 2, sticky = "nsew")
cajaTextoMatrizCor.config(yscrollcommand = scrollMatrizCor.set)


"""
Se definen los botones de la página 5 Cluster
"""
frameBotonesP5 = Frame(p5)
frameBotonesP5.grid(row = 4, column = 0)        


textoKmeans = Label(frameBotonesP5, text = "KMeans")
textoKmeans.grid(row=4, column=0, padx=5,pady=5)
textoClusters = Label(frameBotonesP5, text = "Clusters")
textoClusters.grid(row = 5, column = 0, padx = 5, pady = 5)
rangoCluster = Spinbox(frameBotonesP5, from_=0, to = 100, increment = 1, width=8)
rangoCluster.grid(row = 5, column = 1, padx = 5, pady = 5)
textoColumnas = Label(frameBotonesP5, text = "Ingrese los numeros de columna de las variables a seleccionar(seperar por ,)")
textoColumnas.grid(row = 5, column = 2, padx = 5, pady = 5)
entradaColumnas = Entry(frameBotonesP5)
entradaColumnas.grid(row = 5, column = 3, padx = 4, pady = 4)
botonVSE = Button(frameBotonesP5, text = "Ejecutar nuevas Variables", command = variablesSeleccionadas)
botonVSE.grid(row=5, column=4, padx=5,pady=5)
botonECurve = Button(frameBotonesP5, text = "Elbow Curve", command = grafica_elbowCurve)
botonECurve.grid(row=6, column=0, padx=5,pady=5)
botonKM = Button(frameBotonesP5, text = "Ejecutar KMeans", command = kMeans)
botonKM.grid(row = 6, column = 1, padx = 5, pady = 5)
botonGrafCen = Button(frameBotonesP5, text = "Graficar Centroides 3D", command = graficaCentros)
botonGrafCen.grid(row = 6, column = 2, padx = 5, pady = 5)

miFrameCluster = Frame(p5)
miFrameCluster.grid(row = 5, column = 0)
cajaTextoCluster = Text(miFrameCluster, width = 100, height = 20)
cajaTextoCluster.grid(row = 7, column = 1 ,padx = 5, pady = 5)
scrollCluster = Scrollbar(miFrameCluster, command = cajaTextoCluster.yview)
scrollCluster.grid(row = 7, column = 2, sticky = "nsew")
cajaTextoCluster.config(yscrollcommand = scrollCluster.set)

"""
Se definen los botones de la página 6
"""
frameBotonesP6 = Frame(p6)
frameBotonesP6.grid(row = 4, column = 0)

textoApoyo = Label(frameBotonesP6, text = "Apoyo:")
textoApoyo.grid(row=4, column=0, padx=5,pady=5)
rangoApoyo = Spinbox(frameBotonesP6, from_=0, to = 1, format = '%10.4f', increment = 0.0100, width=8)
rangoApoyo.grid(row=4, column=1, padx=5,pady=5)
textoConfianza= Label(frameBotonesP6, text = "Confianza:")
textoConfianza.grid(row=4, column=2, padx=5,pady=5)
rangoConfianza = Spinbox(frameBotonesP6, from_=0, to = 1, format = '%10.4f', increment = 0.0100, width=8)
rangoConfianza.grid(row=4, column=3, padx=5,pady=5)
textoMinL= Label(frameBotonesP6, text = "Min Length:")
textoMinL.grid(row=4, column=4, padx=5,pady=5)
rangoMinL = Spinbox(frameBotonesP6, from_=0, to = 100, increment = 1, width=5)
rangoMinL.grid(row=4, column=5, padx=5,pady=5)
textoMinLift = Label(frameBotonesP6, text = "Lift")
textoMinLift.grid(row=4, column=6, padx=5,pady=5)
rangoMinLift = Spinbox(frameBotonesP6, from_=0, to = 100, increment = 1, width=5)
rangoMinLift.grid(row=4, column=7, padx=5,pady=5)
botonMosReglas = Button(frameBotonesP6, text = "Mostrar Reglas", command = mostrarReglasAsociacion)
botonMosReglas.grid(row=5, column=0, padx=5,pady=5)
textoSort= Label(frameBotonesP6, text = "Sort by:")
textoSort.grid(row=5, column=1, padx=5,pady=5)
opcionSort = StringVar(frameBotonesP6)
opcionSort.set("Support")
seleccion = OptionMenu(frameBotonesP6, opcionSort, "Support", "Confidence","Lift")
seleccion.grid(row=5, column=2, padx=5,pady=5)
botonDiagrama = Button(frameBotonesP6, text = "Diagrama", command = graficarRA)
botonDiagrama.grid(row=5, column=3, padx=5,pady=5)
miFrameReglas = Frame(p6)
miFrameReglas.grid(row = 5, column=0)
cajaTextoReglas = Text(miFrameReglas, width = 160, height = 20)
cajaTextoReglas.grid(row = 7, column = 1 ,padx = 5, pady = 5)
scrollReglas = Scrollbar(miFrameReglas, command = cajaTextoReglas.yview)
scrollReglas.grid(row = 7, column = 2, sticky = "nsew")
cajaTextoReglas.config(yscrollcommand = scrollReglas.set)

"""Se definen botones de la página 9"""
frameBotonesP9 = Frame(p9)
frameBotonesP9.grid(row = 4, column = 0)
textoMetodoDist = Label(frameBotonesP9, text = "Metodo:")
textoMetodoDist.grid(row=4, column=0, padx=5,pady=5)
opcionMetodDist = StringVar(frameBotonesP9)
opcionMetodDist.set("Euclidean")
seleccionMetodDist = OptionMenu(frameBotonesP9, opcionMetodDist, "Euclidean", "Chebyshev", "Cityblock","Minkowski")
seleccionMetodDist.grid(row=4, column=1, padx=5,pady=5)
botonMatrizDist = Button(frameBotonesP9, text = "Matriz Distancias", command = construyeMatrizDistancias)
botonMatrizDist.grid(row=4, column=2, padx=5,pady=5)

miFrameDistancias = Frame(p9)
miFrameDistancias.grid(row = 5, column = 0)
cajaTextoDistancias = Text(miFrameDistancias, width = 150, height = 20)
cajaTextoDistancias.grid(row = 7, column = 1 ,padx = 5, pady = 5)
scrollDistancias = Scrollbar(miFrameDistancias, command = cajaTextoDistancias.yview)
scrollDistancias.grid(row = 7, column = 2, sticky = "nsew")
cajaTextoDistancias.config(yscrollcommand = scrollDistancias.set)

"""Se definen botones de la página 10"""
frameBotonesP10 = Frame(p10)
frameBotonesP10.grid(row = 4, column = 0)

textoIDPac = Label (frameBotonesP10, text = "Paciente ID:")
textoIDPac.grid(row = 4, column=0, padx = 4, pady = 4)
entradaID = Entry(frameBotonesP10)
entradaID.grid(row = 4, column = 1, padx = 4, pady = 4)

textoTexture = Label (frameBotonesP10, text = "Texture:")
textoTexture.grid(row = 5, column=0, padx = 4, pady = 4)
entradaTexture = Entry(frameBotonesP10)
entradaTexture.grid(row = 5, column = 1, padx = 4, pady = 4)

textoArea = Label (frameBotonesP10, text = "Area:")
textoArea.grid(row = 6, column=0, padx = 4, pady = 4)
entradaArea = Entry(frameBotonesP10)
entradaArea.grid(row = 6, column = 1, padx = 4, pady = 4)

textoCompactness = Label (frameBotonesP10, text = "Compactness:")
textoCompactness.grid(row = 7, column=0, padx = 4, pady = 4)
entradaCompactness = Entry(frameBotonesP10)
entradaCompactness.grid(row = 7, column = 1, padx = 4, pady = 4)

textoConcavity = Label (frameBotonesP10, text = "Concavity:")
textoConcavity.grid(row = 4, column=2, padx = 4, pady = 4)
entradaConcavity = Entry(frameBotonesP10)
entradaConcavity.grid(row = 4, column = 3, padx = 4, pady = 4)

textoSymmetry = Label (frameBotonesP10, text = "Symmetry:")
textoSymmetry.grid(row = 5, column=2, padx = 4, pady = 4)
entradaSymmetry = Entry(frameBotonesP10)
entradaSymmetry.grid(row = 5, column = 3, padx = 4, pady = 4)

textoFD = Label (frameBotonesP10, text = "Fractal Dimension:")
textoFD.grid(row = 6, column=2, padx = 4, pady = 4)
entradaFD = Entry(frameBotonesP10)
entradaFD.grid(row = 6, column = 3, padx = 4, pady = 4)

botonAnalisis = Button(frameBotonesP10, text = "Analizar paciente", command = diagnosticaPaciente)
botonAnalisis.grid(row = 8, column = 0, padx = 5, pady = 5)

miFrameCancer = Frame(p10)
miFrameCancer.grid(row = 5, column = 0)
cajaTextoCancer = Text(miFrameCancer, width = 100, height = 20)
cajaTextoCancer.grid(row = 7, column = 1 ,padx = 5, pady = 5)
scrollCancer = Scrollbar(miFrameCancer, command = cajaTextoCancer.yview)
scrollCancer.grid(row = 7, column = 2, sticky = "nsew")
cajaTextoCancer.config(yscrollcommand = scrollCancer.set)

raiz.mainloop()