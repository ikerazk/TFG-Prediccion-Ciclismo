import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

##########################################################################################################################################################

def crear_dataset(ciclistas):
    ###1.Obtener el año a partir de la fecha
    ciclistas['Date'] = pd.to_datetime(ciclistas['Date'])
    ciclistas['año'] = ciclistas['Date'].dt.year
    
    ###2.Crear un dataset auxiliar con año,nombres,equipos y datos necesarios
    datos_aux = ciclistas[['año','Name','Team','Race Name','Rank','PCS points']].copy()
    
    #Lista de paises que aparecen en el nombre del equipo
    paises = [
        'albania','algeria','argentina','armenia','australia','austria','azerbaijan',
        'belarus','belgium','bolivia','bosnia','brazil','bulgaria','cameroon','canada','chile','china',
        'colombia','costa rica','croatia','cuba','cyprus','czech republic','denmark',
        'dominican republic','ecuador','egypt','el salvador','eritrea','estonia','eslovenia','ethiopia','finland',
        'france','georgia','germany','great britain','greece','guatemala','honduras',
        'hong kong','hungary','iceland','india','indonesia','iran','iraq','ireland',
        'israel','italy','jamaica','japan','jordan','kazakhstan','kosovo','latvia','lebanon','liechtenstein','lithuania',
        'luxembourg','malaysia','malta','mexico','moldova','monaco','mongolia',
        'montenegro','morocco','namibia','netherlands','new zealand','nicaragua',
        'niger','nigeria','north korea','north macedonia','norway','pakistan','panama','paraguay','peru',
        'philippines','poland','portugal','puerto rico','qatar','romania','russia','rwanda','saudi arabia',
        'senegal','serbia','singapore','slovakia','slovenia','south africa','south korea','spain','sweden',
        'switzerland','syria','taiwan','tajikistan','thailand','tunisia','turkey',
        'turkmenistan','uganda','ukraine','united arab emirates','united kingdom','united states','uruguay','uzbekistan',
        'venezuela','vietnam','zimbabwe'
    ]
    #Normalizar para poder comparar nombre de paises
    datos_aux['Team_norm'] = datos_aux['Team'].str.lower().str.strip().copy()
    
    #Filtrar fuera las selecciones nacionales
    datos_aux = datos_aux[~datos_aux['Team_norm'].isin(paises)]
    
    #Renombrar las columnas
    datos_aux = datos_aux.rename(columns = {'Name' : 'nombre','Team_norm' : 'equipo','Race Name' : 'carrera','PCS points' : 'puntos'})
    datos_aux = datos_aux.drop(columns=['Team'])
    
    return datos_aux

##########################################################################################################################################################

def completar_dataset(datos_aux):
    ###3.Obtener los datos individuales de cada ciclista
    datos_aux['Rank'] = pd.to_numeric(datos_aux['Rank'],errors='coerce')
    
    datos_aux['top1'] = (datos_aux['Rank'] == 1).astype(int)
    datos_aux['top3'] = (datos_aux['Rank'] <= 3).astype(int)
    datos_aux['top10'] = (datos_aux['Rank'] <= 10).astype(int)
    
    individual = datos_aux.groupby(['año','nombre'])
    datos_individual = individual.agg(
        num_carreras = ('Rank','count'),
        puntos = ('puntos','sum'),
        top1 = ('top1','sum'),
        top3 = ('top3','sum'),
        top10 = ('top10','sum')).reset_index()
    
    
    ###4.Obtener los datos del equipo en las carreras que ha participado cada ciclista
    ##4.1 Los puntos que ha conseguido cada equipo por carrera
    equipo = datos_aux.groupby(['año','equipo','carrera'])
    equipo_carrera = equipo.agg(
        puntos_equipo = ('puntos','sum'),
        top1_equipo = ('top1','sum'),
        top3_equipo = ('top3','sum'),
        top10_equipo = ('top10','sum')).reset_index()
    
    ##4.2 Unir para cada ciclista sus carrearas con el total del equipo
    ciclista_equipo = pd.merge(
        datos_aux[['año','nombre','equipo','carrera']],
        equipo_carrera,
        on=['año','equipo','carrera'],
        how='left'
    ).drop_duplicates()
    
    ##4.3 Obtener los datos del equpo donde participo el ciclista
    grupo_eq = ciclista_equipo.groupby(['año','nombre'])
    datos_equipos = grupo_eq.agg(
        e_puntos = ('puntos_equipo','sum'),
        e_top1 = ('top1_equipo','sum'),
        e_top3 = ('top3_equipo','sum'),
        e_top10 = ('top10_equipo','sum')
    ).reset_index()
    
    ###5.Fusionar datasets con los datos individuales y los de equipo
    datos_final = pd.merge(datos_individual, datos_equipos, on=['año','nombre',], how='left')

    return datos_final

##########################################################################################################################################################

def añadir_edad(datos_final,df_ciclistas_edad):
    df_ciclistas_edad['nombre_kaggle'] = df_ciclistas_edad['Last Name'].str.strip() + ' ' + df_ciclistas_edad['First Name'].str.strip()
    
    df_ciclistas_edad['nombre_limpio'] = df_ciclistas_edad['nombre_kaggle'].str.lower().str.strip()
    datos_final['nombre_limpio'] = datos_final['nombre'].str.lower().str.strip()
    
    df_ciclistas_edad_limpio = df_ciclistas_edad[['nombre_limpio', 'Birth date']].drop_duplicates(subset=['nombre_limpio'])
    
    datos_final_con_edad = pd.merge(
        datos_final,
        df_ciclistas_edad_limpio,
        on='nombre_limpio', 
        how='left'
    )
    
    total_filas = len(datos_final_con_edad)
    encontrados = datos_final_con_edad['Birth date'].notna().sum()
    no_encontrados = datos_final_con_edad['Birth date'].isna().sum()
    
    print(f"\nTotal de muestras: {total_filas}")
    print(f"Fechas de nacimiento encontradas: {encontrados}")
    print(f"Fechas de nacimiento no encontradas (NaN): {no_encontrados}")
    print(f"Porcentaje de muestras cambiadas: {(encontrados / total_filas) * 100}%")
    
    datos_final_con_edad['fecha_nacimiento'] = pd.to_datetime(datos_final_con_edad['Birth date'], errors='coerce')
    datos_final_con_edad['edad'] = datos_final_con_edad['año'] - datos_final_con_edad['fecha_nacimiento'].dt.year
    
    mediana_edad = datos_final_con_edad['edad'].median()
    
    print(f"\nLa mediana de edad es: {mediana_edad} años")
    
    datos_final_con_edad['edad'] = datos_final_con_edad['edad'].fillna(mediana_edad)
    datos_final_con_edad['edad'] = datos_final_con_edad['edad'].astype(int)
    
    print(datos_final_con_edad[['nombre', 'año', 'fecha_nacimiento', 'edad']].head())
    
    return datos_final_con_edad

##########################################################################################################################################################

def crear_dataset_prediccion(datos_final):
    
    # Columnas para ayudar predecir(t-2,t-1)
    #variables_para_x = ['edad','num_carreras', 'puntos', 'top1', 'top3', 'top10','e_puntos', 'e_top1', 'e_top3', 'e_top10']
    variables_para_x = ['num_carreras', 'puntos', 'top1', 'top3', 'top10','e_puntos', 'e_top1', 'e_top3', 'e_top10']
        
    datos_prediccion = []
    
    for nombre in datos_final['nombre'].unique():
        ciclista_data = datos_final[datos_final['nombre'] == nombre].sort_values(by="año")
        
        if len(ciclista_data) < 3:
            continue
            
        for i in range(2, len(ciclista_data)):
            #año t(actual)
            fila_t = ciclista_data.iloc[i]
            
            #año t-1(anterior)
            fila_t_1 = ciclista_data.iloc[i-1]
            
            #año t-2(hace dos años)
            fila_t_2 = ciclista_data.iloc[i-2]
            
            #comprobar que los 3 años son seguidos
            if (fila_t['año'] - fila_t_1['año'] == 1) and (fila_t_1['año'] - fila_t_2['año'] == 1):
                
                #crear la fila
                nueva_fila = {'nombre': nombre, 'año': fila_t['año']}
                
                #añadir las variables objetivo
                nueva_fila['puntos_t'] = fila_t['puntos']
                nueva_fila['e_puntos_t'] = fila_t['e_puntos']

                carreras_t1 = fila_t_1['num_carreras']
                
                #añadir las variables de años anteriores
                for col in variables_para_x:
                    val_t_1 = fila_t_1[col]
                    val_t_2 = fila_t_2[col]
                    
                    nueva_fila[f"{col}_t-1"] = val_t_1
                    nueva_fila[f"{col}_t-2"] = val_t_2

                    #añadir el ratio de evolucion
                    # if val_t_2 == 0:
                    #     if val_t_1 > 0:
                    #         #ratio_evol = 2.0
                    #         ratio_evol = val_t_1
                    #     else:
                    #         ratio_evol = 0.0
                    # else:
                    #     ratio_evol = val_t_1 / val_t_2
                    # nueva_fila[f"{col}_ratio_evol"] = ratio_evol
                    
                    # #añadir el ratio teniendo en cuenta el numero de carreras en el que ha participado
                    # if carreras_t1 > 0:
                    #     ratio_carrera = val_t_1 / carreras_t1
                    # else:
                    #     ratio_carrera = 0.0
                    # nueva_fila[f"{col}_por_carrera_t-1"] = ratio_carrera
                    
                datos_prediccion.append(nueva_fila)
    
    prediccion_dataset = pd.DataFrame(datos_prediccion)

    return prediccion_dataset

##########################################################################################################################################################

def crear_dataset_prediccion_3_temporadas(datos_final):

    variables_para_x = ['num_carreras', 'puntos', 'top1', 'top3', 'top10','e_puntos', 'e_top1', 'e_top3', 'e_top10']
        
    datos_prediccion = []

    for nombre in datos_final['nombre'].unique():
        ciclista_data = datos_final[datos_final['nombre'] == nombre].sort_values(by="año")

        if len(ciclista_data) < 4:
            continue

        for i in range(3, len(ciclista_data)):
            fila_t = ciclista_data.iloc[i]
            fila_t_1 = ciclista_data.iloc[i-1]
            fila_t_2 = ciclista_data.iloc[i-2]
            fila_t_3 = ciclista_data.iloc[i-3]

            condicion = ((fila_t['año'] - fila_t_1['año'] == 1) and (fila_t_1['año'] - fila_t_2['año'] == 1) and (fila_t_2['año'] - fila_t_3['año'] == 1))
            
            if condicion:
                nueva_fila = {'nombre': nombre, 'año': fila_t['año']}
                
                nueva_fila['puntos_t'] = fila_t['puntos']
                nueva_fila['e_puntos_t'] = fila_t['e_puntos']

                for col in variables_para_x:
                    val_t_1 = fila_t_1[col]
                    val_t_2 = fila_t_2[col]
                    val_t_3 = fila_t_3[col]

                    nueva_fila[f"{col}_t-1"] = val_t_1
                    nueva_fila[f"{col}_t-2"] = val_t_2
                    nueva_fila[f"{col}_t-3"] = val_t_3
                    
                datos_prediccion.append(nueva_fila)

    prediccion_dataset = pd.DataFrame(datos_prediccion)

    return prediccion_dataset
##########################################################################################################################################################

def dividir_dataset(data,año_limite_train,año_limite_val):

    train_data = data[data['año'] <= año_limite_train]
    val_data = data[(data['año'] > año_limite_train) & (data['año'] <= año_limite_val)] 
    test_data = data[data['año'] > año_limite_val] 

    return train_data,val_data,test_data

##########################################################################################################################################################

def generar_conjuntos(train_data,val_data,test_data,X_cols,y_col):
    
    X_train = train_data[X_cols]
    y_train = train_data[y_col]
    X_val = val_data[X_cols]
    y_val = val_data[y_col]
    X_test = test_data[X_cols]
    y_test = test_data[y_col]

    return X_train,y_train,X_val,y_val,X_test,y_test

##########################################################################################################################################################

def dividir_en_conjuntos(train_data,val_data,test_data):

    #todas_nuevas = [col for col in train_data.columns if 'ratio' in col or 'por_carrera' in col]
    #todas_nuevas = [col for col in train_data.columns if  'ratio' in col]
    #todas_nuevas = [col for col in train_data.columns if  'por_carrera' in col]
    #nuevas_equipo = [col for col in todas_nuevas if col.startswith('e_')]
    #nuevas_individuales = [col for col in todas_nuevas if not col.startswith('e_')]

    ###Opcion 1:conocer puntos del ciclista a partir de sus datos

    #X_cols_1 = ['edad_t-1','edad_t-2','puntos_t-2', 'top1_t-2', 'top3_t-2', 'top10_t-2', 'num_carreras_t-2','puntos_t-1', 'top1_t-1', 'top3_t-1', 'top10_t-1', 'num_carreras_t-1']
    # X_cols_1 = [
    #     'puntos_t-3', 'top1_t-3', 'top3_t-3', 'top10_t-3', 'num_carreras_t-3', 
    #     'puntos_t-2', 'top1_t-2', 'top3_t-2', 'top10_t-2', 'num_carreras_t-2',
    #     'puntos_t-1', 'top1_t-1', 'top3_t-1', 'top10_t-1', 'num_carreras_t-1'
    # ]
    X_cols_1 = ['puntos_t-2', 'top1_t-2', 'top3_t-2', 'top10_t-2', 'num_carreras_t-2','puntos_t-1', 'top1_t-1', 'top3_t-1', 'top10_t-1', 'num_carreras_t-1']
    #X_cols_1 = X_cols_1 + nuevas_individuales
    y_col_1 = 'puntos_t'
    
    X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = generar_conjuntos(train_data, val_data, test_data, X_cols_1, y_col_1)
    
    ###Opcion 2:conocer los puntos que aporta al equipo a partir de la informacion del equipo

    # X_cols_2 = [
    #     'e_puntos_t-3', 'e_top1_t-3', 'e_top3_t-3', 'e_top10_t-3', 
    #     'e_puntos_t-2', 'e_top1_t-2', 'e_top3_t-2', 'e_top10_t-2',
    #     'e_puntos_t-1', 'e_top1_t-1', 'e_top3_t-1', 'e_top10_t-1'
    # ]
    X_cols_2 = ['e_puntos_t-2', 'e_top1_t-2', 'e_top3_t-2', 'e_top10_t-2','e_puntos_t-1', 'e_top1_t-1', 'e_top3_t-1', 'e_top10_t-1']
    #X_cols_2 = X_cols_2 + nuevas_equipo
    y_col_2 = 'e_puntos_t'
    
    X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = generar_conjuntos(train_data, val_data, test_data, X_cols_2, y_col_2)
    
    ###Opcion 3:conocer los puntos del ciclista teniendo en cuenta todos los datos
    
    # X_cols_3 = ['edad_t-1','edad_t-2','puntos_t-2', 'top1_t-2', 'top3_t-2', 'top10_t-2', 'num_carreras_t-2',
    #             'e_puntos_t-2', 'e_top1_t-2', 'e_top3_t-2', 'e_top10_t-2',
    #             'puntos_t-1', 'top1_t-1', 'top3_t-1', 'top10_t-1', 'num_carreras_t-1',
    #             'e_puntos_t-1', 'e_top1_t-1', 'e_top3_t-1', 'e_top10_t-1']
    # X_cols_3 = [
    #     'puntos_t-3', 'top1_t-3', 'top3_t-3', 'top10_t-3', 'num_carreras_t-3',
    #     'e_puntos_t-3', 'e_top1_t-3', 'e_top3_t-3', 'e_top10_t-3',             
    #     'puntos_t-2', 'top1_t-2', 'top3_t-2', 'top10_t-2', 'num_carreras_t-2',
    #     'e_puntos_t-2', 'e_top1_t-2', 'e_top3_t-2', 'e_top10_t-2',
    #     'puntos_t-1', 'top1_t-1', 'top3_t-1', 'top10_t-1', 'num_carreras_t-1',
    #     'e_puntos_t-1', 'e_top1_t-1', 'e_top3_t-1', 'e_top10_t-1'
    # ]
    X_cols_3 = ['puntos_t-2', 'top1_t-2', 'top3_t-2', 'top10_t-2', 'num_carreras_t-2',
                'e_puntos_t-2', 'e_top1_t-2', 'e_top3_t-2', 'e_top10_t-2',
                'puntos_t-1', 'top1_t-1', 'top3_t-1', 'top10_t-1', 'num_carreras_t-1',
                'e_puntos_t-1', 'e_top1_t-1', 'e_top3_t-1', 'e_top10_t-1']
    #X_cols_3 = X_cols_3 + todas_nuevas
    y_col_3 = 'puntos_t'
    
    X_train_3, y_train_3, X_val_3, y_val_3, X_test_3, y_test_3 = generar_conjuntos(train_data, val_data, test_data, X_cols_3, y_col_3)
    
    ###Opcion 4:conocer los puntos que aporta al equipo teniendo en cuenta todos los datos
    X_cols_4 = X_cols_3
    y_col_4 = 'e_puntos_t'
    
    X_train_4, y_train_4, X_val_4, y_val_4, X_test_4, y_test_4 = generar_conjuntos(train_data, val_data, test_data, X_cols_4, y_col_4)

    datos = [
        (X_train_1, y_train_1, X_val_1, y_val_1,X_test_1,y_test_1),
        (X_train_2, y_train_2, X_val_2, y_val_2,X_test_2, y_test_2),
        (X_train_3, y_train_3, X_val_3, y_val_3,X_test_3, y_test_3),
        (X_train_4, y_train_4, X_val_4, y_val_4,X_test_4, y_test_4)
    ]

    return datos

##########################################################################################################################################################

def calcular_accuracy_within_n(y_real, y_pred, tam_grupos=10, margen_error=1):
    
    datos = pd.DataFrame({'real': y_real, 'pred': y_pred})

    datos['grupo_real'] = pd.qcut(datos['real'].rank(method='first'), q=tam_grupos, labels=False)
    datos['grupo_pred'] = pd.qcut(datos['pred'].rank(method='first'), q=tam_grupos, labels=False)

    datos['acierto'] = ((datos['grupo_pred'] >= datos['grupo_real'] - margen_error) & (datos['grupo_pred'] <= datos['grupo_real'] + margen_error))
    
    return datos['acierto'].mean()

##########################################################################################################################################################

def evaluar_modelos(X_train, y_train, X_val, y_val,modelos):
    resultados = {}
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train,y_train)
        y_pred = modelo.predict(X_val)
    
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        spearmanr_corr, _ = spearmanr(y_val, y_pred) 
        acc_within_1 = calcular_accuracy_within_n(y_val, y_pred, tam_grupos=10, margen_error=1) * 100
    
        indices_no_cero = y_val > 0
            
        if np.any(indices_no_cero):
            mape = mean_absolute_percentage_error(y_val[indices_no_cero], y_pred[indices_no_cero]) * 100
        else:
            mape = np.inf
    
        resultados[nombre] = {
            "MAE": mae,
            "MAPE": mape,
            "MSE": mse,
            "R2": r2,
            "Spearmanr": spearmanr_corr,
            "Acc_Within_1": acc_within_1
        }
    grafico_real_vs_predicho(y_val,y_pred)
    resultados = pd.DataFrame(resultados).T
    resultados = resultados.sort_values(by="Spearmanr",ascending=False)
    resultados = resultados.round(2)
    
    return resultados

##########################################################################################################################################################

def realizar_prediccion(datos,modelos):

    X_train_1, y_train_1, X_val_1, y_val_1 = datos[0]
    X_train_2, y_train_2, X_val_2, y_val_2 = datos[1]
    X_train_3, y_train_3, X_val_3, y_val_3 = datos[2]
    X_train_4, y_train_4, X_val_4, y_val_4 = datos[3]
    
    print("Opcion 1:conocer puntos del ciclista a partir de sus datos\n")
    resultados_1 = evaluar_modelos(X_train_1, y_train_1, X_val_1, y_val_1,modelos)
    resultados_1.to_csv("resultados_opcion1.csv",sep=';', decimal=',', index=True)
    print(resultados_1)
    print("\n\n")
    
    print("Opcion 2:conocer los puntos que aporta al equipo a partir de los puntos del equipo\n")
    resultados_2 = evaluar_modelos(X_train_2, y_train_2, X_val_2, y_val_2,modelos)
    resultados_2.to_csv("resultados_opcion2.csv",sep=';', decimal=',', index=True)
    print(resultados_2)
    print("\n\n")
    
    print("Opcion 3:conocer los puntos del ciclista teniendo en cuenta todos los datos\n")
    resultados_3 = evaluar_modelos(X_train_3, y_train_3, X_val_3, y_val_3,modelos)
    resultados_3.to_csv("resultados_opcion3.csv",sep=';', decimal=',', index=True)
    print(resultados_3)
    print("\n\n")
    
    print("Opcion 4:conocer los puntos que aporta al equipo teniendo en cuenta todos los datos\n")
    resultados_4 = evaluar_modelos(X_train_4, y_train_4, X_val_4, y_val_4,modelos)
    resultados_4.to_csv("resultados_opcion4.csv",sep=';', decimal=',', index=True)
    print(resultados_4)
    print("\n\n")

    return resultados_1, resultados_2, resultados_3, resultados_4

##########################################################################################################################################################

def spearman_score_func(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred)
    return corr

##########################################################################################################################################################

def ejecutar_grid_search(modelo, params, X, y, nombre, objetivo):
    print(f"OPTIMIZANDO: {nombre}")
    print(f"Objetivo: Maximizar {objetivo}")

    spearman_scorer = make_scorer(spearman_score_func, greater_is_better=True)
    
    if objetivo == 'Spearman':
        scoring = spearman_scorer
        metrica = "Spearman"
    else:
        scoring = 'neg_mean_absolute_error'
        metrica = "MAE"
    
    grid = GridSearchCV(
        estimator=modelo,
        param_grid=params,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    # Mostrar resultados
    mejor_score = grid.best_score_
    if objetivo == 'MAE':
        mejor_score = -mejor_score 
        
    print(f"Mejores parámetros: {grid.best_params_}")
    print(f"Mejor {metrica} (Validación Cruzada): {mejor_score:.4f}")
    print("-" * 50)
    
    return grid.best_estimator_

##########################################################################################################################################################

def conteo_uci_vs_pcs(ciclistas):
    total_muestras = len(ciclistas)
    n_uci = len(ciclistas[ciclistas['UCI points'] > 0])
    n_pcs = len(ciclistas[ciclistas['PCS points'] > 0])
    
    barras = plt.bar(['Puntos UCI', 'Puntos PCS'], [n_uci, n_pcs], color=['blue', 'orange'])
    
    plt.title(f'Resultados Puntuables (Total Muestras: {total_muestras})')
    plt.ylabel('Número de carreras con puntos')
    
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2, altura, f'{altura}', ha='center', va='bottom')
    plt.show()

##########################################################################################################################################################

def grafico_recuento_puntos(nombre_archivo):
    
    data = pd.read_csv(nombre_archivo, sep=';', decimal=',')
    
    con_puntos = data[data['puntos_t'] > 0]
    con_cero_puntos = data[data['puntos_t'] == 0]
    
    recuento_cero = len(con_cero_puntos)
    
    bins = [1, 50, 100, 150, 200, 250,300,350,400,450,500, np.inf]
    labels = ['1-50', '50-100', '100-150', '150-200', '200-250', '250-300','300-350','350-400','400-450','450-500','500+']
    
    recuento_rangos = pd.cut(con_puntos['puntos_t'], bins=bins, labels=labels, right=False).value_counts().sort_index()

    df_recuentos = pd.DataFrame({'Cantidad': [recuento_cero]}, index=['0 Puntos'])
    
    df_recuentos = pd.concat([df_recuentos, recuento_rangos.to_frame(name='Cantidad')])
    
    plt.figure(figsize=(12, 7)) 
    
    ax = df_recuentos['Cantidad'].plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title('Grafico recuento de puntos')
    plt.xlabel('Rango de Puntos')
    plt.ylabel('Num. de Filas')
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout() 
    plt.show() 

##########################################################################################################################################################

def grafico_real_vs_predicho(y,y_pred):
    
    df_resultados = pd.DataFrame({
        'real': y,
        'predicho': y_pred
    })
            
    plt.figure(figsize=(10, 8))
    
    plt.scatter(df_resultados['real'], df_resultados['predicho'], alpha=0.3)
    
    limite = df_resultados['real'].max()
    plt.plot([0, limite], [0, limite], 'r--', linewidth=2, label='Predicción Perfecta (y=x)')
    
    plt.title('Real vs Predicho')
    plt.xlabel('Puntos Reales')
    plt.ylabel('Puntos Predichos')
    plt.legend()
    plt.grid(True) 
    plt.xlim(0, 2000) 
    plt.ylim(0, 2000)
    plt.show()

##########################################################################################################################################################

def grafico_errores(y,y_pred):
    
    df_resultados = pd.DataFrame({
        'real': y,
        'predicho': y_pred
    })
    
    df_resultados['error'] = df_resultados['real'] - df_resultados['predicho']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_resultados['real'], df_resultados['error'], alpha=0.3)
    
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Error Cero')
    
    plt.title('Grafico de Errores')
    plt.xlabel('Puntos Reales')
    plt.ylabel('Error (Real - Predicho)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 2000)
    plt.show()

##########################################################################################################################################################