# %% ============== celda 1
#!/usr/bin/python
# -*- coding: utf-8 -*-

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
#    "dummies". Una veraz leído y transformado el archivo a .fea cambie a falso
#    la variable CARGA para facilitar la ejecución del código conforme desarrolle
#    su examen.
# 3) El código contempla otras variables booleanas que determinan la salida o no
#    de gráficos  y trazas de diferentes tópicos. Los nombres de las variables
#    indican la salida supervisada.
# 4) El caso que se trata es el de "los Impresionistas", ya trabajado anteriormente,
#    Note que la variavle objetivo es "BAND". dado que ambas partes de la muestra
#    interesan, llamaremos:
#    band capturados  y no band incurridos a los casos respectivos
#    con score  superior o igual al considerado y
#    no band determinados y band perdidos a los que tienen score menor al
#    de corte considerado.      
#
# 5) Al finalizar su trabajo o cuando el tiempo se agote suba su código a CANVAS.
#
#******************************************************************************
from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors                import ListedColormap
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
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                           random_state=1, n_clusters_per_class=1)
#rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
#linearly_separable = (X, y)

#datasets = [make_moons(noise=0.3, random_state=0),
#            make_circles(noise=0.2, factor=0.5, random_state=1),
#            linearly_separable
#            ]
# =============================================================================
#   Variables para la localización del archivo de datos
# =============================================================================
disco        = 'C:'
tray        = r'C:\Users\dvdve\OneDrive - INSTITUTO TECNOLOGICO AUTONOMO DE MEXICO\Semestre 10\Minería de datos\Parcial 2\\'
# tray         = disco + '/user/materias/MD/MD_202101/Casos/Impresionistas/'

nomArchDatos = 'bandingData.arff'
# =============================================================================
#
# =============================================================================
#
#ds=pd.read_csv(tray + 'bandingData.csv')

Carga            = False
GRAFICA_ROC_PRC  = True
GRAFICA_CU       = False
GRAFICA_CUMSUMVO = False
TRAZA            = False

if Carga:
    data     = arff.loadarff(tray + nomArchDatos)
    datos = pd.DataFrame(data[0])
    datos.fillna(0,inplace=True)
    for c in list(datos.columns):
           if TRAZA:
              print('...................'+c)
           datos[c] = datos.apply(lambda r: decodifica(r[c]),axis=1)
           datos[c] = datos[c].astype(str).str.lower()
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
     
    ls_convert = ['grain_screened',
                  'ink_color',
                  'proof_on_ctd_ink',
                  'blade_mfg',
                  'cylinder_division',
                  'paper_type',
                  'ink_type',
                  'direct_steam',
                  'solvent_type',
                  'type_on_cylinder',
                  'press_type',
                  'press',
                  'unit_number',
                  'cylinder_size',                  
                  'paper_mill_location',
                  'plating_tank',
                  'band_type'
                  ]
    for v in ls_convert:
        convierte(datos,v)
        
    datos.drop(labels=ls_convert,axis=1,inplace=True)
    
    if TRAZA:
      print(cadsep)
      print(' ===================> Post Conversión <====================')
      print(cadsep)
      for k,c in enumerate(datos.columns): print(k,c)
      print(cadsep)
    
    # ================= Para evitar problemas =====================
    
    datos=datos.astype(float)
    
    datos.to_feather(tray + 'datosB.fea')
    if datos.isna().any().sum() > 0:
        salir('No todos los datos se han homogeneizado')
else:
    datos=pd.read_feather(tray + 'datosB.fea')
    

#salir('Datos convertidos')    
# =============================================================================
#                          Datos Homogenizados
# =============================================================================
datasets = [datos]

VO = 'band_type_band'

c_u_band_as_band       =   0.5
c_u_band_as_no_band    = - 2.5
c_u_no_band_as_no_band =   3.5
c_u_no_band_as_band    =   0.5



# *****************************************************************************
#  Agregue aquí lo que requiera para el ejercicio 4 (ver más adelante)
# *****************************************************************************

# iterando sobre el conjunto de datos
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    lsc = list(ds.columns)
    lsc.remove(VO)
    X = ds[lsc]
    y = ds[VO]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    print_id()   
    #
    # *************************************************************************    
    # [2.5] EJERCICIO 1) Considere las submuestras train y test
    #
    # 1.1) Obtenga los casos totales y BAND (VO) y NO BAND por cada submuestra
    # 1.2) Obtenga las densidades originales por cada submuestra
    # 1.3) Obtenga la Utilidad Máxima posible por submuestra
    # *************************************************************************    
    #
    casos_train         = len(X_train)
    casos_band_train    = len(y_train[y_train==True])
    casos_no_band_train = len(y_train[y_train==False])

    casos_test          = len(y_test)
    casos_band_test     = len(y_test[y_test==True])
    casos_no_band_test  = len(y_test[y_test==False])
   
    densidad_band_train    = casos_band_train/casos_train
    densidad_no_band_train = casos_no_band_train/casos_train

    densidad_band_test     = casos_band_test/casos_test
    densidad_no_band_test  = casos_no_band_test/casos_test
    
    UMP_train              = casos_band_train * c_u_band_as_band + casos_no_band_train * c_u_no_band_as_no_band
    UMP_test               = casos_band_test * c_u_band_as_band + casos_no_band_test * c_u_no_band_as_no_band
    
    print(cadsep)    
    print('Ejercicio 1):')
    print(cadsep)
    print('           casos_train:', casos_train)
    print('      casos_band_train:', casos_band_train)
    print('   casos_no_band_train:', casos_no_band_train)
    print('   densidad_band_train:', '{:6.4f}'.format(densidad_band_train))
    print('densidad_no_band_train:', '{:6.4f}'.format(densidad_no_band_train))
    print('             UMP train:', '{:12,.2f}'.format(UMP_train))
    print()
    print('            casos_test:', casos_test)
    print('       casos_band_test:', casos_band_test)
    print('    casos_no_band_test:', casos_no_band_test)
    print('    densidad_band_test:', '{:6.4f}'.format(densidad_band_test))
    print(' densidad_no_band_test:', '{:6.4f}'.format(densidad_no_band_test))
    print('              UMP test:', '{:12,.2f}'.format(UMP_test))
    print(cadsep)
    # *************************************************************************
    #   FIN EJERCICIO 1)
    # *************************************************************************
    #
    # iterando sobre los clasificadores
    UC_max = 0
    name_max = ''
    CAP_max = 0
    for name, clf in zip(names, classifiers):
        print(cadsep)
        print('Modelando con ' + name)
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
        #
        # por lo pronto solamente tiene estos valores de u y c
        #
        u = 780
        c = 400
        
        preds['UPC']    = preds.apply(lambda row: (u if row['truelabel']==1 else -c),axis=1)
        preds['UPAcum'] = preds.UPC.cumsum()

        #print(preds['prediction'].value_counts())
        if type(clf) is DecisionTreeClassifier:
            pt=preds.pivot_table(values="UPC",index="prediction",columns='truelabel',aggfunc="count",fill_value= 0)
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

        positive_data = preds[preds['num_caso']<=CAP]
        negative_data = preds[preds['num_caso']>CAP]

        Num_BAND_Capt        = len(positive_data[positive_data['truelabel']==True])
        Num_NO_BAND_Inc      = len(positive_data[positive_data['truelabel']==False])
        densidad_band_capt   = Num_BAND_Capt/len(positive_data)
        lift_band_capt       = densidad_band_capt-densidad_band_test

        Num_NO_BAND_Det      = len(negative_data[negative_data['truelabel']==False])
        Num_BAND_Per         = len(negative_data[negative_data['truelabel']==True])
        densidad_no_band_det = Num_NO_BAND_Det/len(negative_data)
        lift_no_band_det     = densidad_no_band_det-densidad_no_band_test
        
        score_de_corte       = sc
        print(cadsep)
        print('Ejercicio 2) ' + name )
        print(cadsep)
        print('                          AROC:{:13.3f}'.format(areaUnderROC))
        print('                          APRC:{:13.3f}'.format(average_precision))
        print('                           UMP:{:12,.2f}'.format(UMP_test))
        print('                            UC:{:12,.2f}'.format(UC))
        print('                           CAP:',CAP)
        print('              Num    BAND Capt:',Num_BAND_Capt)
        print('              Num NO BAND  Inc:',Num_NO_BAND_Inc)
        print('         Densidad de BAND Capt:',densidad_band_capt)
        print('             Lift    BAND Capt:',lift_band_capt)
        print()
        print('              Num NO BAND  Det:',Num_NO_BAND_Det)
        print('              Num    BAND  Per:',Num_BAND_Per)
        print('      Densidad de NO BAND  Det:',densidad_no_band_det)
        print('             Lift NO BAND  Det:',lift_no_band_det)
        print(cadsep)


        if UC>UC_max:
            UC_max = UC
            name_max = name
            CAP_max = CAP
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
        prop_capt = Num_BAND_Capt/(Num_BAND_Capt+Num_BAND_Per)
        prop_inc  = Num_NO_BAND_Inc/(Num_NO_BAND_Inc+Num_NO_BAND_Det)
        #
        # PRC
        #
        prec_capt = Num_BAND_Capt/(Num_BAND_Capt+Num_NO_BAND_Inc)
        rec_capt = Num_BAND_Capt/(Num_BAND_Capt+Num_BAND_Per)
        #
        # *********************************************************************
        #    FIN DEL EJERCICIO 3)
        # *********************************************************************
        #
        #  ================ codigo original
        # if GRAFICA_ROC_PRC:
        #     plt.figure(figsize=(12,5))
        #     plt.subplot(121)                                                              
        #     plt.step(recall,precision,color='k',alpha=0.07,where='post')
        #     plt.plot(prop_inc,prop_capt,'or')
        #     plt.fill_between(recall,precision,step='post',alpha=0.3,color='k')
        #     plt.xlabel('Recall')
        #     plt.ylabel('Precision')
        #     plt.ylim([0,1.05])
        #     plt.xlim([0,1.05])
        #     plt.title(name + '\nPrecision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))
                   
        #     plt.subplot(122)
        #     plt.plot(fpr,tpr,color='r',lw=2,label='ROC curve')
        #     plt.plot(rec_capt,prec_capt,'or')
        #     plt.plot([0,1],[0,1],color = 'k',lw = 2,linestyle='--')
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')
        #     plt.xlim([0,1])
        #     plt.ylim([0,1.05])
        #     plt.title(name + '\nReceiver Operating characteristic: Area under the curve:{0:0.2f}'.format(areaUnderROC))
        #     plt.legend(loc='lower right')
        #     plt.show()


        if GRAFICA_ROC_PRC:
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

print(cadsep)
# 
# *****************************************************************************
# [2.5] EJERCICIO 4) Imprima nombre del mejor modelo, su UC y su CAP
# *****************************************************************************            
mejor_modelo = name_max
UC_mejor_modelo = UC_max
CAP_mejor_modelo = CAP_max

print(cadsep)
print("El mejor modelo es:" + mejor_modelo)
print('UC:{:12,.2f}'.format(UC_mejor_modelo))
print('CAP mejor modelo:', CAP_mejor_modelo)
print(cadsep)        
print_id()
print('                     ' + str_dateTime)        
print(cadsep)




