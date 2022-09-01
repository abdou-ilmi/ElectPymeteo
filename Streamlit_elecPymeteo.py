#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:33:14 2021

@author: igal
"""

import streamlit as st

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import seaborn as sns
from prophet import Prophet
import statsmodels.api as sm 
import itertools
import datetime
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


import scipy.stats as stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


#Création de differents partie
st.sidebar.title('Projet elecPymeteo')
st.sidebar.subheader('Menu')
dif_parti=["Dataviz régionale","Prévision de la consommation (série temporelle)", \
           "Prévision de la consommation (données météorologiques)",'Prévision de la production éolienne']
Partie=st.sidebar.radio(' ',options=dif_parti)

#Partie=st.selectbox('choix paragraphe', options=dif_parti)


#st.write('Partie: ', Partie)
df=pd.read_csv('donnee_regional.csv', sep=';', parse_dates=[3], squeeze=True, usecols=[0,1,2,3,4,5,6,7,8,9])
#st.write(df.head())
df=df.rename(columns={'Consommation (MW)':'Consommation'})

df=df.drop(['Code INSEE région','Nature','Date','Heure','Date - Heure'],axis=1)

df_Bour_FC=df[(df.Région=='Bourgogne-Franche-Comté')]
df_ile_d_fr=df[(df.Région=='Île-de-France')]
date= pd.date_range("2013-01-01", periods=148944, freq="0.5H")
df_ile_d_fr['date']=date
df_Bour_FC['date']=date

df_ile_defrance=df_ile_d_fr[['date','Consommation']]
df_ile_defrance=df_ile_defrance.set_index('date')
df_ile_defrance=df_ile_defrance.fillna(method='bfill')
df_ile_defrance_heure=df_ile_defrance.resample('H').sum()
df_ile_defrance_mois=df_ile_defrance.resample('M').sum()

#prepartion de données de la region Pays de la loir pour la previon de la production
df=df.rename(columns={'Eolien (MW)':'Eolien'})

df_P_d_loire=df[(df.Région=='Pays de la Loire')]

df_P_d_loire['date']=date

df_pays_d_loire=df_P_d_loire[['date','Eolien']]
df_pays_d_loire=df_pays_d_loire.set_index('date')
df_pays_d_loire=df_pays_d_loire.fillna(method='bfill')

df_pays_d_loire_3h=df_pays_d_loire.resample('3H').sum()
df_pays_de_loire_2020=df_pays_d_loire_3h['Eolien'].loc['2020-01-01 00:00:00':'2021-06-30 21:00:00']

df_geoloc=pd.read_csv('Géolocalisation_éoliennes.csv')

df_geoloc=df_geoloc.rename(columns={'Distance à la station (m)':'distance'})

df_vent_pdl=pd.read_csv('Vent_PDL.csv',sep=',',parse_dates=[0], squeeze=True)

df_vent_pdl=df_vent_pdl.replace('mq','0')

df_vent_pdl=df_vent_pdl.rename(columns={'Date - Heure':'date'})

coef_puissance=[]
for k in range(len(df_geoloc['distance'])):
    if df_geoloc['distance'][k]>50000:
        coef_puissance.append(0.001)
    else:
        coef=(-0.89*df_geoloc['distance'][k]+44950)/45000
        coef_puissance.append(coef)
df_geoloc['coef_puissance']=coef_puissance



vent_vitesse_pondere=[]

for k in range(len(df_vent_pdl)):
        som_vent=0.158*float(df_vent_pdl.iloc[k,1])+0.223*float(df_vent_pdl.iloc[k,2])+0.088*float(df_vent_pdl.iloc[k,3])+0.184*float(df_vent_pdl.iloc[k,4])
        vent_vitesse_pondere.append( som_vent/4)
df_vent_pdl['vent_vitesse_pondere']=vent_vitesse_pondere

df_vent_pdl=df_vent_pdl.set_index('date')

data_pays_loire_reg=df_vent_pdl.merge(right = df_pays_de_loire_2020, on = 'date', how = 'inner')

y=data_pays_loire_reg['Eolien']
X=data_pays_loire_reg['vent_vitesse_pondere']


y=pd.DataFrame(y)
X=pd.DataFrame(X)

train_size_eo=int(len(X) *0.8)
test_size_eo=int(len(X))-train_size_eo
X_train_eo=X[:train_size_eo]
X_test_eo=X[train_size_eo:]
y_train_eo=y[:train_size_eo]
y_test_eo=y[train_size_eo:]

y_nor=pd.DataFrame(preprocessing.StandardScaler().fit_transform(y))
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_size=int(len(df_ile_defrance_mois) *0.7)
test_size=int(len(df_ile_defrance_mois))-train_size
X_train=df_ile_defrance_mois[:train_size]
X_test=df_ile_defrance_mois[train_size:]

df_ile_defrance_pro_M=df_ile_defrance_mois.rename(columns={'Consommation':'y'})
df_ile_defrance_pro_M=df_ile_defrance_pro_M.reset_index('date')
df_ile_defrance_pro_M=df_ile_defrance_pro_M.rename(columns={'date':'ds'})


train_size_p=int(len(df_ile_defrance_pro_M) *0.7)
test_size_p=int(len(df_ile_defrance_pro_M))-train_size_p
X_train_pro=df_ile_defrance_pro_M[:train_size_p]
X_test_pro=df_ile_defrance_pro_M[train_size_p:]

df_Bour_FC_conso=df_Bour_FC[['date','Consommation']]
df_Bour_FC_conso=df_Bour_FC_conso.set_index('date')

df_Bour_FC_conso=df_Bour_FC_conso.fillna(method='bfill')
df_Bour_FC_conso_heure=df_Bour_FC_conso.resample('H').sum()

df_Bour_FC_conso_jour=df_Bour_FC_conso.resample('D').sum()


df_Bour_FC_conso['year']=df_Bour_FC_conso.index.year
df_Bour_FC_conso['Month']=df_Bour_FC_conso.index.month
df_Bour_FC_conso['day']=df_Bour_FC_conso.index.day
df_Bour_FC_conso['hour']=df_Bour_FC_conso.index.hour
df_Bour_FC_conso['month_name']=df_Bour_FC_conso.index.month_name()
df_Bour_FC_conso['day_name']=df_Bour_FC_conso.index.day_name()

som_de_conso_par_jour=df_Bour_FC_conso.groupby('day_name').sum().Consommation.reset_index()

som_de_conso_par_mois=df_Bour_FC_conso.groupby('month_name').sum().Consommation.reset_index()

som_de_conso_par_heure=df_Bour_FC_conso.groupby('hour').sum().Consommation.reset_index()




if Partie==dif_parti[0]:
    st.title("Consommation électrique en Bourgogne-Franche-Comté")
    st.info("Dans cette section, nous proposons quelques datavisualisations permettant, à l'échelle d'une région, \
             de confirmer le caractère multipériodique de la consommation électrique.")
    options=['Consommation/jour','Consommation/mois','Consommation/heure', 'Profil de consommation selon la saison']
    choix=st.selectbox('Type de graphique', options=options)
    st.write('Graphique choisi : ', choix)
    
    if choix==options[0]:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(8, 4))
        order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        sns.barplot(x=som_de_conso_par_jour['day_name'], y=som_de_conso_par_jour['Consommation'], order=order)
        plt.xlabel('jour')
        plt.title('consommation par rapport aux jours')
        st.pyplot();
    elif choix==options[1]:
        plt.figure(figsize=(11, 6))
        order_m=['January','February','March','April','May','June','July','September','October','November','December']
        sns.barplot(x=som_de_conso_par_mois['month_name'], y=som_de_conso_par_mois['Consommation'], order=order_m)
        plt.xlabel('mois')
        plt.title('consommation par rapport aux mois')
        st.pyplot();
    elif choix==options[2]: 
        plt.figure(figsize=(10, 6))
        sns.barplot(x=som_de_conso_par_heure['hour'], y=som_de_conso_par_heure['Consommation'])
        plt.xlabel('heure')
        plt.title('consommation par rapport aux heures')
        st.pyplot();
    else:
        plt.figure(figsize=(15, 6))
        plt.subplot(1,2,1)
        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2013-01-01 00:00:00':'2013-01-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2013-01-01 00:00:00':'2013-01-07 23:00:00'],
         label='hiver')
        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2013-04-01 00:00:00':'2013-04-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2013-04-01 00:00:00':'2013-04-07 23:00:00'],
         label='Printemps')
        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2013-07-01 00:00:00':'2013-07-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2013-07-01 00:00:00':'2013-07-07 23:00:00'],
         label='été')

        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2013-10-01 00:00:00':'2013-10-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2013-10-01 00:00:00':'2013-10-07 23:00:00'],
         label='automne')

        plt.xlabel('Heure')
        plt.title('Consommation de 2013')
        plt.legend();
        plt.subplot(1,2,2)


        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2019-01-01 00:00:00':'2019-01-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2019-01-01 00:00:00':'2019-01-07 23:00:00'],
         label='hiver')
        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2019-04-01 00:00:00':'2019-04-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2013-04-01 00:00:00':'2013-04-07 23:00:00'],
         label='Printemps')
        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2019-07-01 00:00:00':'2019-07-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2019-07-01 00:00:00':'2019-07-07 23:00:00'],
         label='été')

        plt.plot(range(len(df_Bour_FC_conso_heure['Consommation'].loc['2019-10-01 00:00:00':'2019-10-07 23:00:00']))
         ,df_Bour_FC_conso_heure['Consommation'].loc['2019-10-01 00:00:00':'2019-10-07 23:00:00'],
         label='automne')

        plt.xlabel('Heure')
        plt.title('Consommation de 2019')
        plt.legend()
        plt.suptitle("Consommation d'une semaine type de chaque saison en 2013 et 2019",fontsize=20);
        st.pyplot();

elif Partie==dif_parti[1]:
    st.title("Prévision à l'aide des modèles traitant les séries temporelle")
    st.info("La consommation électrique présente un caractère multipériodique (annuel, hebdomadaire et quotidien). Nous pouvons \
            donc utiliser les modèles de prévision communément utilisés pour modéliser les séries temporelles.")
    Actuelle=pd.DataFrame(df_ile_defrance_mois.iloc[train_size:, 0])
    options=['SARIMA','Prophet']
    choix=st.selectbox('Choix du modèle', options=options)
    st.write('Modèle choisi : ', choix)
    if choix==options[0]:
#SARIMA(0,1,1)(1,1,0,12)
        model_manuelle_6= sm.tsa.statespace.SARIMAX(X_train, 
                                    order=(0,1,1), 
                                    seasonal_order=(1,1,1,12), 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False,
                                    freq='M')
                                    
# Fit the model
        resultat_manuelle_6=model_manuelle_6.fit()
    
# Resumé de Resultats
   # print(resultat_manuelle_6.summary())
        y_pred_m6=resultat_manuelle_6.predict(start =train_size, end=train_size+test_size-1)

        pred_int_conf_m6=resultat_manuelle_6.get_prediction(start=train_size, end=train_size+test_size-1)

        int_conf_m6=pred_int_conf_m6.conf_int()
        y_pred_m6=pd.DataFrame(y_pred_m6)
        y_pred_m6.reset_index(drop=True, inplace=True)
        y_pred_m6.index=X_test.index
        y_pred_m6['valeur_réelle'] = Actuelle['Consommation']
        y_pred_m6.columns = ['valeur_predite','valeur_réelle']
        #y_pred_m6.rename(columns=0:'valeur_predite'}, inplace=True) 
        mape_m6=mean_absolute_percentage_error(y_pred_m6['valeur_réelle'],y_pred_m6['valeur_predite'])
        
        #st.write("l'erreur moyenne absolue du modèle est :", mape_m6)
        st.write("L'erreur moyenne absolue sur les données de test est de :", round(mape_m6,2), "%")
        st.text('Les graphiques suivants representent les valeurs observées et les valeurs prédites\n par le modèle SARMA')
        pred_m6=resultat_manuelle_6.predict(102, 124)#Prédiction
        df_pred_m6 = pd.concat([df_ile_defrance_mois, pred_m6])#Concaténation des prédictions
        pred_m6=resultat_manuelle_6.predict(102, 124)#Prédiction
        df_pred_m6 = pd.concat([df_ile_defrance_mois, pred_m6])
        fig = plt.figure(figsize=(20,8))
        plt.subplot(221)

        plt.plot(y_pred_m6['valeur_réelle'], color='blue', label='valeurs réelles')
        plt.plot(y_pred_m6['valeur_predite'], color='red', label='valeurs predite')
        plt.fill_between(int_conf_m6.index[1:],
                int_conf_m6.iloc[1:, 0],
                int_conf_m6.iloc[1:, 1], color='k', alpha=.2)
        plt.ylabel('consommation')
        plt.legend()
        plt.title('Valeurs actuelles et prédites avec intervalle de confiance de 95%');
        plt.subplot(222)

        plt.plot(y_pred_m6['valeur_réelle'], color='blue', label='valeurs réelles')
        plt.plot(y_pred_m6['valeur_predite'], color='red', label='valeurs predite')
        plt.ylabel('consommation')
        plt.legend()
        plt.title('Valeurs actuelles et prédites')
        plt.subplot(223)

    
        plt.plot(df_pred_m6) #Visualisation

        plt.title('previsions à horizon 24');
        plt.axvline(x= df_pred_m6.index[len(X_train)+len(X_test)], color='red'); # Ajout de la ligne verticale
        st.pyplot();
        plt.subplot(223)
    else:
        model_prophet = Prophet()
        model_prophet.fit(X_train_pro)
        future = model_prophet.make_future_dataframe(periods=31, freq='M')
        futur_forecast = model_prophet.predict(df=future)
        y_pred_prophet=futur_forecast['yhat']
        y_pred_prophet=pd.DataFrame(y_pred_prophet)
        y_pred_prophet.reset_index(drop=True, inplace=True)
        y_pred_prophet=y_pred_prophet.rename(columns={'yhat':'valeur_predite'})
        y_pred_prophet['ds']=df_ile_defrance_pro_M.ds
        y_pred_prophet['valeur_réelle'] =df_ile_defrance_pro_M['y']
        y_pred_prophet['yhat_upper']=futur_forecast['yhat_upper'].values
        y_pred_prophet['yhat_lower']=futur_forecast['yhat_lower'].values
        y_pred_prophet=y_pred_prophet.rename(columns={'ds':'date'}) 
        y_pred_prophet=y_pred_prophet.set_index('date')
        mape_pro1=mean_absolute_percentage_error(y_pred_prophet['valeur_réelle'],y_pred_prophet['valeur_predite'])
        
        #st.write("l'erreur moyenne absolue du modèle est :", mape_pro1)
        st.write("L'erreur moyenne absolue sur les données de test est de :", round(mape_pro1,2), "%")
        
        st.text('Les graphiques suivants representent les valeurs observées et les valeurs prédites\n par le modèle Prophet')
        fig = plt.figure(figsize=(15,6))
        plt.subplot(121)
        y_pred_prophet['valeur_réelle'].plot(figsize=(15,8), legend=True, color='blue', label='valeurs réelles')
        y_pred_prophet['valeur_predite'].plot(legend=True, color='red', figsize=(15,8), label='valeurs predite')
        plt.ylabel('consommation')
        plt.legend()
        plt.title('Valeurs actuelles et prédites');
        plt.subplot(122)
        plt.plot(y_pred_prophet['valeur_predite'], label='valeurs prédites')
        plt.plot(y_pred_prophet['valeur_réelle'], label='valeurs actuelles')

        plt.fill_between(y_pred_prophet['yhat_lower'].index,
                y_pred_prophet['yhat_upper'],
                y_pred_prophet['yhat_lower'], color='b', alpha=.2)
#plt.plot(y_pred_prophet['yhat_upper'],'--', label='valeurs prédites maximum')
#plt.plot(y_pred_prophet['yhat_lower'],'--',label='valeurs prédites minimum');
        plt.xlabel('date')
        plt.ylabel('consommation')
        plt.title(' Valeurs prédites et actuelles avec intervalle de confiance')
        plt.legend()
        st.pyplot();


elif Partie==dif_parti[2]:
    st.title("Prévision de la consommation grâce à des variables externes")
    st.info("La consommation électrique a un caractère périodique correspondant à l'organisation de notre société industrielle \
            mais dépend également fortement de la température extérieure. Nous allons ici prédire la consommation à l'aide de \
            modèles de régression s'appuyant sur ces 2 composantes.")


    st.subheader("Consommation en fonction de la température")

    ###   PREPARATION DES DONNEES METEO   ###
    
    #On prépare un dataframe unique à partir des dataframes mensuels récoltés
    df_meteo = pd.read_csv("Données_météo_complètes.csv", parse_dates = [2])
    
    df_meteo_idf = df_meteo[df_meteo["numer_sta"]==7149].loc[:,["date","t"]]
    df_meteo_idf.columns = ["Date - Heure", "Température (K)"]
    
    ### PREPARATION DES DONNEES ENERGETIQUES   ###
    
    données = pd.read_csv("donnee_regional.csv", sep =";")
    
    #A partir du dataframe des données, on nettoie les cellules vides de la colonne "Consommation (MW)" on met la colonne
    #temporelle au même format que le dataframe météo et on isole les données de la région IDF  
    def preprocessing_ener_idf(df):
        df = df[df["Consommation (MW)"].notna()]
        df['Date - Heure'] = df['Date - Heure'].apply(lambda x: x[:-6])
        df['Date - Heure'] = pd.to_datetime(df['Date - Heure'], format = "%Y-%m-%dT%H:%M:%S")
        df = df.sort_values(by = ["Date - Heure"])
        df = df[df["Région"]=="Île-de-France"].loc[:,["Date - Heure","Consommation (MW)"]]
        return df
    
    df_energ_idf = données.copy()
    df_energ_idf = preprocessing_ener_idf(df_energ_idf)
    
    df_complet = df_energ_idf.merge(right = df_meteo_idf, on = "Date - Heure", how = "inner")
    df_complet["Jour de la semaine"] = df_complet["Date - Heure"].dt.weekday
    df_complet["Heure"] = df_complet["Date - Heure"].dt.hour
    df_complet["Mois"] = df_complet["Date - Heure"].dt.month
    df_complet = df_complet.set_index('Date - Heure')
    df_complet = df_complet.drop(df_complet[df_complet["Température (K)"]=='mq'].index,axis=0)
    df_complet["Température (K)"] = df_complet["Température (K)"].astype(float)
    df = df_complet["2013":"2019"] #on va établir notre modèle sur le période précédant la crise COVID
    
    #VISUALISATION DE LA CONSOMMATION PAR RAPPORT A LA TEMPERATURE
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(df['Température (K)'], df['Consommation (MW)'], color='darkblue')
    ax1.set_xlabel('Température (K)')
    ax1.set_ylabel('Consommation (MW)')
    st.pyplot(fig1)
    
    ###########################################################################################
    st.subheader("Modélisation")
    data = df.loc[:,['Jour de la semaine','Mois','Heure', 'Température (K)']]
    target = df['Consommation (MW)']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)
    
    
    modèles = ("RandomForestRegressor","KNeighborsRegressor","Ridge","Lasso")
    choix = st.selectbox("Choix du modèle :", options = modèles)
    
    st.write("Modèle sélectionné :", choix)
    
    
    def train_model(choix):
        if choix == modèles[0]:
            model = RandomForestRegressor(bootstrap= True,
             max_depth= 50,
             max_features= 'sqrt',
             min_samples_leaf= 2,
             min_samples_split= 15,
             n_estimators= 700)
        elif choix == modèles[1]:
            nb_voisins = st.slider("n_neighbors :", 2, 14, step = 2)
            model = KNeighborsRegressor(n_neighbors = nb_voisins)
        elif choix == modèles[2]:
            model = RidgeCV()
        else :
            model = Lasso()
        model.fit(X_train, y_train)
        df_complet["Prédiction de consommation (MW)"] = model.predict(df_complet[['Jour de la semaine','Mois','Heure', 'Température (K)']])
        fig2 = plt.figure(figsize=(15,10))
        ax2 = fig2.add_subplot(111)
        ax2.plot(df_complet.loc[:,'Consommation (MW)'].resample("W").mean(), label = 'Consommation (MW)')
        ax2.plot(df_complet.loc[:,'Prédiction de consommation (MW)'].resample("W").mean(), label = 'Prédiction de consommation (MW)')
        ax2.legend()
        ax2.set_xlabel("Temps")
        ax2.set_ylabel("Consommation (MW)")
        ax2.set_title("Consommation lissée par semaine (2012 à 2021)")
        y_pred = model.predict(X_test)
        mape = np.mean(np.abs((y_test - y_pred)/ y_test))*100
        return mape, fig2
    
    mape, fig2 = train_model(choix)
    
    st.write("L'erreur moyenne absolue sur les données de test est de :", round(mape,2), "%")
    st.pyplot(fig2)

    



else:
    st.title("Prévision de la production éolienne")
    st.info("Tout comme nous pouvons prédire la consommation électrique d'une région à l'aide de la température, nous pouvons \
            prévoir la production électrique éolienne d'une région à l'aide de la vitesse du vent")


    st.subheader("Implantation géographique des éoliennes et des stations météorologiques utilisées")
    
    
    df = pd.read_csv("Géolocalisation_éoliennes.csv")

    nantes = [47.15, -1.61, "Nantes"]
    alencon = [48.45, 0.11, "Alençon"]
    tours = [47.44, 0.73, "Tours"]
    la_rochelle = [46.04, -1.405, "La Rochelle"]
    
    #CREATION DE LA CARTE
    import folium
    import json
    from folium.plugins import BeautifyIcon
    from streamlit_folium import folium_static
    
    lat_centre = 47.4  
    long_centre = -0.9   
    zoom = 7
    carte= folium.Map(location=[lat_centre,long_centre],zoom_start=zoom)
    
    for i in range(df.shape[0]):
        latitude = df.iloc[i,1]
        longitude = df.iloc[i,2]
        puissance = df.iloc[i,3]
        m = "<strong>" + "Puissance : " + "</strong>" + str(puissance) + " MW"\
        + "<br>Station la plus proche : " + df.iloc[i,5]
        folium.CircleMarker([latitude, longitude], radius = 5,color=None,fill_color ="red",
                        fill_opacity=0.5,popup = folium.Popup(m, max_width = 400)).add_to(carte)
    
    folium.Marker([nantes[0],nantes[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = nantes[2]).add_to(carte)
    folium.Marker([alencon[0], alencon[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = alencon[2]).add_to(carte)
    folium.Marker([tours[0], tours[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = tours[2]).add_to(carte)
    folium.Marker([la_rochelle[0], la_rochelle[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = la_rochelle[2]).add_to(carte)
    
    folium_static(carte)

    st.subheader("Production éolienne en fonction de la vitesse du vent")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pays_loire_reg['vent_vitesse_pondere'],data_pays_loire_reg['Eolien'],color='darkblue');
    plt.xlabel('Vitesse de vent')
    plt.ylabel("Production de l'éolien (kW)")
    #plt.title('Nuage de points entre production Eolienne et vitesse de vent');
    st.pyplot();
    
    st.subheader("Modélisation")
    
    options=['LinearRegression','Ridge','Lasso']
    choix=st.selectbox('Choix du modèle', options=options)
    st.write('Modèle choisi : ', choix)
    

    if choix==options[0]:
        model=LinearRegression()
    elif choix==options[1]:
        alphas_value=st.slider('alpha',min_value=0.01, max_value=8.0, step=.01)
        model=Ridge(alpha=alphas_value)
    else :
        alphas_value=st.slider('alpha',min_value=0.01, max_value=1.0, step=.01)
        model=Lasso(alpha=alphas_value)
      
    model.fit(X_train_eo, y_train_eo)
    st.write("Le score du modèle d'entraiment est :",   round(model.score(X_train_eo, y_train_eo),2))
    st.write('Le score du modèle de test  est :',   round(model.score(X_test_eo, y_test_eo),2))   
    pred_test = model.predict(X_test_eo)
    pred_test=pd.DataFrame(pred_test)
    pred_test=pred_test.rename(columns={0:'valeur_predite'})
    y_prevision=pred_test
    new_index=y_test_eo.index
    y_prevision=y_prevision.set_index(new_index)
    y_prevision['valeur_observé']=y_test_eo['Eolien']
    y_pred_semaine=y_prevision.resample('D').sum()
    fig = plt.figure(figsize=(12,6))
    plt.plot(y_pred_semaine['valeur_predite'], label='valeurs prédites')
    plt.plot(y_pred_semaine['valeur_observé'], label='valeurs observées')
    plt.xlabel('date')
    plt.ylabel('Production (kW)')
    plt.legend()
    plt.title('Production Éolienne')
    st.pyplot();