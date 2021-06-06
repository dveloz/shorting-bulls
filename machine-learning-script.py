#!/usr/bin/python
# -*- coding: utf-8 -*-
# %%
"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
#print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
# 
#******************************************************************************
#   Segundo examen parcial de Minería de Datos, Grupo 02, 
#                 semestre enero - mayo de 2021
# Comentarios e instrucciones:
# 1) Asigne su CU y Nombre en los lugares correspondientes;
# 2) Asigne las variables de disco y tray para la lectura del archivo de datos,
#    en este caso se proporciona una archivo en formato .arff y se incluye el
#    código para leerlo y transformar las variables nominales a indicadoras o 
#    "dummies". Una vez leído y transformado el archivo a .fea cambie a falso
#    la variable CARGA para facilitar la ejecución del código conforme desarrolle
#    su examen.
# 3) El código contempla otras variables booleanas que determinan la salida o no
#    de gráficos  y trazas de diferentes tópicos. Los nombres de las variables
#    indican la salida supervisada.
# 4) El caso que se trata es el de "los Impresionistas", ya trabajado anteriormente,
#    Note que la variable objetivo es "VO_1", dado en ocasiones que ambas partes de la muestra
#    interesan, llamaremos:
#    VO_1 capturados  y VO_0 incurridos a los casos respectivos
#    con score  superior o igual al considerado y
#    VO_0 determinados y VO_1 perdidos a los que tienen score menor al
#    de corte considerado.      
#
# 5) Al finalizar su trabajo o cuando el tiempo se agote suba su código a CANVAS.
#
#******************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors                import ListedColormap
from numpy.core.numeric import True_
from sklearn.model_selection          import train_test_split
from sklearn.preprocessing            import StandardScaler
from sklearn.datasets                 import make_moons, make_circles, make_classification
from sklearn.neural_network           import MLPClassifier
from sklearn.neighbors                import KNeighborsClassifier
from sklearn.svm                      import SVC
from sklearn.gaussian_process         import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree                     import DecisionTreeClassifier
from sklearn.ensemble                 import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes              import GaussianNB
from sklearn.discriminant_analysis    import QuadraticDiscriminantAnalysis

from sklearn.metrics import  precision_recall_curve, average_precision_score
from sklearn.metrics import  roc_curve,auc,roc_auc_score
from sklearn.metrics import  confusion_matrix,classification_report


from scipy.io import arff
import pandas as pd
import sys
import time
import datetime

cadsep = '='*60
t0 = time.time()
# =============================================================================
#   Asigne su CU y nombre como cadenas de caracteres
# =============================================================================
CU     = '163678'
NOMBRE = 'VELOZ SOLORZANO DAVID'

def print_id():
    global CU
    global NOMBRE
    print(cadsep)
    print('CU:' + CU + ' ... ' + NOMBRE)
    print(cadsep)
# =============================================================================
#                                     salir
# =============================================================================
def salir(letrero):
    global t0
    print(cadsep)
    print(letrero)
    t1 = time.time()
    deltaT = t1-t0
    print('DeltaT:','{:8.3} segs.'.format(deltaT))
    print(cadsep)
    sys.exit(0)

# =============================================================================
#                                 convierte    
# =============================================================================

def convierte(df,var):
    valores = df[var].unique()[:-1]
    for valor in valores:
        df[var + '_' + str(valor)] = df.apply(lambda r: 1*(str(r[var])==valor),axis=1)

def decodifica(valor):
     if type(valor) is bytes:
         val = valor.decode('utf-8')
     else:
         val = valor
     return val
 
# =============================================================================
# =============================================================================
#                               PROGRAMA PRINCIPAL
# =============================================================================
# =============================================================================
str_dateTime   = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
print(cadsep)
print('                  COM-23106 Minería de Datos')
print('              semestre enero - mayo 2021, Gpo 02')
print('              Segundo examen Parcial: 3 de mayo de 2021')
print(' '*25 + str_dateTime)
print(cadsep)
names = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         #"Gaussian Process",
         "Decision Tree", 
         "Random Forest", 
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# =============================================================================
#   Variables para la localización del archivo de datos
# =============================================================================

tray         = r'C:\Users\dvdve\OneDrive - INSTITUTO TECNOLOGICO AUTONOMO DE MEXICO\Semestre 10\Minería de datos\parcial2_comp_163678' + '\\'
nomArchDatos = 'Tic2000AllData.xlsx'
# =============================================================================
#
# =============================================================================
#
#ds=pd.read_csv(tray + 'bandingData.csv')

Carga            = False
GRAFICA_ROC_PRC  = False
GRAFICA_CU       = False
GRAFICA_CUMSUMVO = False
TRAZA            = False

#
# lista para el Ejercicio 4
#

lis_res_modelos = []


if Carga:
    #data     = arff.loadarff(tray + nomArchDatos)  
    #datos = pd.DataFrame(data[0])
    datos = pd.read_excel(tray+ nomArchDatos)
    datos.fillna(0,inplace=True)
    for c in list(datos.columns):
           if TRAZA:
              print('...................'+c)
           datos[c] = datos.apply(lambda r: decodifica(r[c]),axis=1)
           datos[c] = datos[c].astype(str).str.upper()
    # =========================================================================
    #  Se reemplazan los '?' por la palabra missing
    # =========================================================================
    datos.replace({'?':'missing'}, inplace=True)
    
    #datos.drop(['timestamp','cylinder_number','customer','job_number'],inplace=True,axis=1)
    if TRAZA:
      print(cadsep)
      print('Antes de la conversión:')
      for k,c in enumerate(datos.columns): print(k,c)
      print(cadsep)
      print('ls_convert = [')
      for c in datos.columns:
        print("'" + c + "',")
      print(']')          
      print(cadsep)
    #
    # =========================================================================
    #    Las siguientes son las variables a ser convertidas en indicadoras
    # ======================= (editar en caso necesario) ======================
    # =========================================================================
     
    #ls_convert = ['BRANCH','RES']
    # for v in ls_convert:
    #     convierte(datos,v)
        
    # datos.drop(labels=ls_convert,axis=1,inplace=True)
    
    # if TRAZA:
    #   print(cadsep)
    #   print(' ===================> Post Conversión <====================')
    #   print(cadsep)
    #   for k,c in enumerate(datos.columns): print(k,c)
    #   print(cadsep)
    
    # ================= Para evitar problemas =====================
    
    datos=datos.astype(float)
    
    datos.to_feather(tray + 'datosINS.fea')
    if datos.isna().any().sum() > 0:
        salir('No todos los datos se han homogeneizado')
else:
    datos=pd.read_feather(tray + 'datosINS.fea')

#salir('Datos convertidos')    
# =============================================================================
#                          Datos Homogenizados
# =============================================================================
datasets = [datos]

VO = 'CARAVAN'

c_u_VO_1_as_1 =  930
c_u_VO_1_as_0 =  0 
c_u_VO_0_as_0 =  0
c_u_VO_0_as_1 =  -70


# *****************************************************************************
#  Agregue aquí lo que requiera para el ejercicio 4 (ver más adelante)
# *****************************************************************************

lis_res_modelos = []

# iterando sobre el conjunto de datos
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    lsc = list(ds.columns)
    lsc.remove(VO)
    X = ds[lsc]
    y = ds[VO]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.45, random_state=42)
    print_id()   
    #
    # *************************************************************************    
    # [2.5] EJERCICIO 1) Considere las submuestras train y test
    #
    # 1.1) Obtenga los casos totales BAND (VO) y NO BAND por cada submuestra
    # 1.2) Obtenga las densidades originales por cada submuestra
    # 1.3) Obtenga la Utilidad Máxima posible por submuestra
    # *************************************************************************    
    #
    casos_train         = y_train.shape[0]
    casos_VO_1_train    = y_train.sum()
    casos_VO_0_train    = casos_train - casos_VO_1_train

    casos_test          = y_test.shape[0]
    casos_VO_1_test     = y_test.sum()
    casos_VO_0_test     = casos_test - casos_VO_1_test
   
    densidad_VO_1_train = casos_VO_1_train / casos_train
    densidad_VO_0_train = 1 - densidad_VO_1_train

    densidad_VO_1_test  = casos_VO_1_test / casos_test
    densidad_VO_0_test  = 1 - densidad_VO_1_test
    
    UMP_train           = casos_VO_1_train * c_u_VO_1_as_1 + casos_VO_0_train * c_u_VO_0_as_0
    UMP_test            = casos_VO_1_test  * c_u_VO_1_as_1 + casos_VO_0_test  * c_u_VO_0_as_0
    
    print(cadsep)    
    print('Ejercicio 1):')
    print(cadsep)
    print('           casos_train:', casos_train)
    print('      casos_VO_1_train:', casos_VO_1_train)
    print('      casos_VO_0_train:', casos_VO_0_train)
    print('   densidad_VO_1_train:', '{:6.4f}'.format(densidad_VO_1_train))
    print('   densidad_VO_0_train:', '{:6.4f}'.format(densidad_VO_0_train))
    print('             UMP train:', '{:12,.2f}'.format(UMP_train))
    print()
    print('            casos_test:', casos_test)
    print('       casos_Vo_1_test:', casos_VO_1_test)
    print('       casos_VO_0_test:', casos_VO_0_test)
    print('    densidad_VO_1_test:', '{:6.4f}'.format(densidad_VO_1_test))
    print('    densidad_VO_0_test:', '{:6.4f}'.format(densidad_VO_0_test))
    print('              UMP test:', '{:12,.2f}'.format(UMP_test))
    print(cadsep)


    # *************************************************************************
    #   FIN EJERCICIO 1)
    # *************************************************************************

    #salir('Se han fijado los parámetros de c/u')   
    
    # iterando sobre los clasificadores
    num_clasif = 0
    for name, clf in zip(names,classifiers): #zip(['Decision Tree'],[DecisionTreeClassifier(max_depth=5)]): #zip(names, classifiers):
        num_clasif += 1
        print(cadsep)
        print(num_clasif, ' ... Modelando con ' + name)
        clf.fit(X_train, y_train)

        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
        else:
            Z = clf.predict_proba(X_test)
        #
        # Evaluación de métricas adicionales para el modelo
        #                         
        preds = pd.DataFrame(y_test)
        if Z.ndim==1:
            s_prob = Z
        else:
            s_prob = Z[:,1]

        preds['prediction'] = s_prob
        
        preds.columns=['truelabel','prediction']
        
        precision,recall,thresholds = precision_recall_curve(preds['truelabel'],preds['prediction'])
        
        average_precision = average_precision_score(preds['truelabel'],preds['prediction'])

        fpr,tpr,thresholds = roc_curve(preds['truelabel'],preds['prediction'])
        
        areaUnderROC = auc(fpr,tpr)
                 
        #
        # Obteniendo la curva de utilidad del modelo
        #
        preds.sort_values(by=['prediction'],ascending = False,inplace=True) # Va por band
        #
        preds['num_caso'] = range(1,preds.shape[0]+1)
        #
        if GRAFICA_CUMSUMVO:
           plt.figure()   
           plt.plot(range(1,preds.shape[0]+1),preds['truelabel'].cumsum())
           plt.xlabel('número de casos')
           plt.ylabel('Cantidad acumulada de positivos')
           plt.title('Curva de acumulado para ' + VO + '\n' + name)
           plt.show()
           
        #
        # Agregue aquí la función de utilidad para cada registro de ser este el corte
        # Destaque el caso del CART
        preds['VO_1_Acum'] = preds['truelabel'].cumsum()
        preds['VO_0_Acum'] = preds.apply(lambda r:r['num_caso'] - r['VO_1_Acum'],axis=1)
        #
        
        preds['UPAcum'] = preds.apply(lambda row:
                          c_u_VO_1_as_1 * row['VO_1_Acum'] +
                          c_u_VO_0_as_1 * row['VO_0_Acum'] +
                          c_u_VO_1_as_0 * ( casos_VO_1_test - row['VO_1_Acum']) +
                          c_u_VO_0_as_0 * ( casos_VO_0_test - row['VO_0_Acum']),
                                      axis= 1)                               
        #
        # Obteniendo la Utilidad capturada
        if name in ['Decision Tree','RBF SVM']:
            u = 930
            c = 70
            pt=preds.pivot_table(values="UPAcum",index="prediction",columns='truelabel',aggfunc="count",fill_value= 0)
            pt.sort_index(ascending=False,inplace=True)
            pt['UPxNodo'] =  -c*pt[0.0] + u*pt[1.0]
            pt['UPAcumxNodo'] = pt['UPxNodo'].cumsum()
            UC = pt.UPAcumxNodo.max()
            sc = pt.index[pt['UPAcumxNodo']==UC].values[0]
            preds['num_caso']=range(1,preds[preds.columns[0]].size+1)
            CAP = (preds.loc[preds['prediction']==sc].num_caso).values.max()
        else:
            UC = preds.UPAcum.max()
            CAP = (preds.loc[preds['UPAcum']==UC].num_caso).values[0]
            sc = (preds.loc[preds['UPAcum']==UC].prediction).values[0]


        #
        
        #
        # *********************************************************************
        # [2.5] EJERCICIO 2)
        #
        # Determine la UC, el CAP,
        # Número de Band Capturados      (o Verdaderos Positivos), 
        # Número de No_Band incurridos   (o Falsos Positivos),
        # Número de NO_BAND determinados (o Verdaderos Negativos),
        # Número de BAND perdidos        (o Falsos Negativos)
        # y la densidades y lift correspondientes
        # *********************************************************************
        #
        Num_VO_1_Capt      = preds[preds['num_caso'] <= CAP]['truelabel'].sum()
        Num_VO_0_Inc       = CAP - Num_VO_1_Capt
        densidad_VO_1_capt = Num_VO_1_Capt / CAP
        lift_VO_1_capt     = densidad_VO_1_capt / densidad_VO_1_test

        
        Num_VO_0_Det       = casos_VO_0_test - Num_VO_0_Inc
        Num_VO_1_Per       = casos_VO_1_test    - Num_VO_1_Capt
        densidad_VO_0_det  = Num_VO_0_Det / (casos_test - CAP)
        lift_VO_0_det      = densidad_VO_0_det / densidad_VO_0_test
        
        score_de_corte     =  sc

        print(cadsep)
        print('Ejercicio 2) ' + name )
        print(type(clf))

        print(cadsep)
        print('                          AROC:{:13.3f}'.format(areaUnderROC))
        print('                          APRC:{:13.3f}'.format(average_precision))
        print('                           UMP:{:12,.2f}'.format(UMP_test))
        print('                            UC:{:12,.2f}'.format(UC))
        print('                           CAP:',CAP)
        print('              Num    VO=1 Capt:',Num_VO_1_Capt)
        print('              Num    VO=0  Inc:',Num_VO_0_Inc)
        print('         Densidad de VO=1 Capt:',densidad_VO_1_capt)
        print('             Lift    VO=1 Capt:',lift_VO_1_capt)
        print()
        print('              Num    VO=0  Det:',Num_VO_0_Det)
        print('              Num    VO=1  Per:',Num_VO_1_Per)
        print('      Densidad de    VO=0  Det:',densidad_VO_0_det)
        print('             Lift    VO=0  Det:',lift_VO_0_det)
        print('                score de corte:',score_de_corte)
        print(cadsep)
        # *********************************************************************
        #  FIN EJERCICIO 2)
        # *********************************************************************
        if GRAFICA_CU:
            plt.figure(figsize=(12,5))
            plt.plot(preds['num_caso'],preds['UPAcum'])
            plt.plot(CAP,UC,'or')
            plt.xlabel('casos')
            plt.ylabel('U_P_acum')
            plt.title(name + '\nCurva de Utilidad')
            plt.show()
 
        # *********************************************************************
        # [2.5] EJERCICIO 3) Determine los puntos de operación en ROC y PRC
        # *********************************************************************
        # ROC
        #
        prop_capt = Num_VO_1_Capt   / casos_VO_1_test
        prop_inc  = Num_VO_0_Inc / casos_VO_0_test
        #
        # PRC
        #
        prec_capt = Num_VO_1_Capt / CAP
        rec_capt  = prop_capt
        #
        # *********************************************************************
        #    FIN DEL EJERCICIO 3)
        # *********************************************************************
        #
        
        if GRAFICA_ROC_PRC:
            #
            #  PRC
            #
            plt.figure(figsize=(12,5))
            plt.subplot(121)                                                              
            plt.step(recall,precision,color='k',alpha=0.07,where='post')
            plt.plot(rec_capt,prec_capt,'or')
            plt.fill_between(recall,precision,step='post',alpha=0.3,color='k')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0,1.05])
            plt.xlim([0,1.05])
            plt.title(name + '\nPrecision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))
            #
            # ROC
            #
            plt.subplot(122)
            plt.plot(fpr,tpr,color='r',lw=2,label='ROC curve')
            plt.plot(prop_inc,prop_capt,'or')           
            plt.plot([0,1],[0,1],color = 'k',lw = 2,linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([0,1])
            plt.ylim([0,1.05])
            plt.title(name + '\nReceiver Operating characteristic: Area under the curve:{0:0.2f}'.format(areaUnderROC))
            plt.legend(loc='lower right')
            plt.show()
       #
       # **********************************************************************
       #  Agregue aquí el código que requiera para el Ejercicio 4) 
       # **********************************************************************
    # Densidad original
    # Densidad de umbral
    # Utilidad máxima posible
    # Utilidad capturada
    # % UC / UMP
    # Número de casos a promover
    # Cantidad de 1’s capturados
    # Cantidad de 0’s incurridos
    # Densidad de captura
    # Lift de captura
    # Costo de la campaña
    # ROI no financiero
    # Score de corte


        lis_res_modelos.append((name,UC,CAP,lift_VO_1_capt, Num_VO_1_Capt, score_de_corte))
print(cadsep)
# 
# *****************************************************************************
# [2.5] EJERCICIO 4) Imprima nombre del mejor modelo, su UC y su CAP
# *****************************************************************************            

# Densidad original
    # Densidad de umbral
    # Utilidad máxima posible
    # Utilidad capturada
    # % UC / UMP
    # Número de casos a promover
    # Cantidad de 1’s capturados
    # Cantidad de 0’s incurridos
    # Densidad de captura
    # Lift de captura
    # Costo de la campaña
    # ROI no financiero
    # Score de corte



for k,t in enumerate(lis_res_modelos):
    if k == 0:
        nameModelo = t[0]
        UCmax      = t[1]
        CAPmax     = t[2]
        liftmax    = t[3]
        num1_max   = t[4]
        score_max  = t[5]
    if t[1] > UCmax:
        nameModelo = t[0]
        UCmax      = t[1]
        CAPmax     = t[2]
        liftmax    = t[3]
        num1_max   = t[4]
        score_max  = t[5]
        
mejor_modelo      = nameModelo
UC_mejor_modelo   = UCmax
uc_ump = UC_mejor_modelo/UMP_test
CAP_mejor_modelo  = CAPmax
lift_mejor_modelo = liftmax

print(cadsep)
print("El mejor modelo es:" + mejor_modelo)
print('                UC:{:12,.2f}'.format(UC_mejor_modelo))
print('               CAP:{:12,.0f}'.format(CAP_mejor_modelo))
print('   lift de captura:{:12,.2f}'.format(lift_mejor_modelo)) 
print(cadsep)        
print('a) Densidad original: {}'.format(densidad_VO_1_test))
print('b) Densidad de umbral: {}'.format(930./1000))
print('c) Utilidad máxima posible: {}'.format(UMP_test))
print('d: Utilidad Capturada: {}'.format(UC_mejor_modelo))
print('e) % UC/UMP: {0:.0%}'.format(uc_ump))
print('f) Número de casos a procesar: {}'.format(CAPmax))
print("g) Cantidad de 1's capturados: {}".format(num1_max))
print("h) Cantidad de 0's incurridos {}".format(CAP_mejor_modelo-num1_max))
print('i) Densidad de captura: {}'.format(num1_max/CAP_mejor_modelo))
print('j) Lift de captura: {}'.format(lift_mejor_modelo))
print('k) Costo de la campaña: {}'.format((CAP_mejor_modelo-num1_max) * 70))
print('l) ROI no financiero: {}'.format(UC_mejor_modelo/((CAP_mejor_modelo-num1_max)*70)))
print('m) Score de corte: {}'.format(score_max))
print(cadsep)
print('Si el modelo se aplicara a 250,000 clientes potenciales (suponiendo mismas densidades que en conjunto test): ')
p = 250000
u = 930
c = -70
print('a) UMP: {}'.format(densidad_VO_1_test*u*p))
nueva_densidad_1 = lift_mejor_modelo*densidad_VO_1_test
nueva_densidad_0 = 1 - nueva_densidad_1
densidad_CAP = CAP_mejor_modelo/casos_train
print('b) UC: {}'.format(p*densidad_CAP*(u*nueva_densidad_1+c*nueva_densidad_0)))
print('c) Costo de la campaña: {}'.format(p*densidad_CAP*nueva_densidad_0*abs(c)))


print_id()
print('                     ' + str_dateTime)        
print(cadsep)






# %%
