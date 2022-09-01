import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import os

###   PREPARATION DES DONNEES METEO   ###

#On prépare un dataframe unique à partir des dataframes mensuels récoltés
big_frame = pd.DataFrame()
for file in os.listdir():
    if file.endswith('.csv'):
        df = pd.read_csv(file, sep=";")
        big_frame = big_frame.append(df, ignore_index=True)

#A partir du dataframe global des données météo, on isole les données météo de la région IDF et on ne conserve que les
#colonnes utiles
def preprocessing_meteo_idf(df):
    df = df[df["numer_sta"]==7149] #on ne garde que les données de la station météo d'Orly
    df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d%H%M%S")
    df = df.loc[:,["date","t"]]
    df.columns = ["Date - Heure", "Température (K)"]
    return df

df_meteo_idf = big_frame.copy()
df_meteo_idf = preprocessing_meteo_idf(df_meteo_idf)


### PREPARATION DES DONNEES ENERGETIQUES   ###

données = pd.read_csv("C:\\DataScientest\\eco2mix-regional-cons-def.csv", sep =";")

#A partir du dataframe des données, on nettoie les cellules vides de la colonne "Consommation (MW)" on met la colonne
#temporelle au même format que le dataframe météo et on isole les données de la région IDF  
def preprocessing_ener_idf(df):
    df = df[df["Consommation (MW)"].notna()]
    df['Date - Heure'] = df['Date - Heure'].apply(lambda x: x[:-6])
    df['Date - Heure'] = pd.to_datetime(df['Date - Heure'], format = "%Y-%m-%dT%H:%M:%S")
    df = df.sort_values(by = ["Date - Heure"])
    df = df[df["Région"]=="Île-de-France"]
    return df

df_energ_idf = données.copy()
df_energ_idf = preprocessing_ener_idf(df_energ_idf)


#CREATION DU DATAFRAME AVEC DONNEES ENERGETIQUES ET DONNEES METEO   ###
df_complet = df_energ_idf.merge(right = df_meteo_idf, on = "Date - Heure", how = "inner")
df_complet["Jour de la semaine"] = df_complet["Date - Heure"].dt.weekday
df_complet["Heure"] = df_complet["Date - Heure"].dt.hour
df_complet["Mois"] = df_complet["Date - Heure"].dt.month
df_complet = df_complet.set_index('Date - Heure')
df_complet = df_complet.drop(df_complet[df_complet["Température (K)"]=='mq'].index,axis=0)
df_complet["Température (K)"] = df_complet["Température (K)"].astype(float)
df = df_complet["2013":"2019"] #on va établir notre modèle sur le période précédant la crise COVID


#VISUALISATION DE LA CONSOMMATION PAR RAPPORT A LA TEMPERATURE   ###
plt.figure(figsize=(10, 8))
plt.scatter(df['Température (K)'], df['Consommation (MW)'], color='darkblue')
plt.xlabel('Température (K)')
plt.ylabel('Consommation (MW)')
plt.show()


### MACHINE LEARNING   ###

data = df.loc[:,['Jour de la semaine','Mois','Heure', 'Température (K)']]
target = df['Consommation (MW)']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)
reg = RandomForestRegressor(n_jobs=-1)

#Recherche des meilleurs paramètres :
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

grid_reg = GridSearchCV(reg, param_grid = random_grid)
grille = grid_reg.fit(X_train, y_train)

print(grid_reg.best_params_)

#Entraînement du modèle avec les meilleurs paramètres
reg_best = RandomForestRegressor(bootstrap= True,
 max_depth= 50,
 max_features= 'sqrt',
 min_samples_leaf= 2,
 min_samples_split= 15,
 n_estimators= 700)
reg_best.fit(X_train, y_train)

print("Score :", reg_best.score(X_train,y_train))
print("Score :", reg_best.score(X_test,y_test))


#DATAVIZ' CONSOMMATION REELLE VS PREDICTIONS   ###

df["Prédiction de consommation (MW)"] = reg_best.predict(df[['Jour de la semaine','Mois','Heure', 'Température (K)']])

fig2 = plt.figure(figsize=(15,10))
ax2 = fig2.add_subplot(111)
ax2.plot(df.loc[:,'Consommation (MW)'].resample("W").mean(), label = 'Consommation (MW)')
ax2.plot(df.loc[:,'Prédiction de consommation (MW)'].resample("W").mean(), label = 'Prédiction de consommation (MW)')
ax2.legend()
ax2.set_xlabel("Temps")
ax2.set_ylabel("Consommation (MW)")
ax2.set_title("Consommation lissée par semaine (2012 à 2021)");


#####################################################################################################################

#MACHINE LEARNING A PARTIR DE LA SERIE TEMPORELLE (CONSOMMATION)

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
from dateutil.relativedelta import relativedelta
import seaborn as sns
#from prophet import Prophet
import statsmodels.api as sm 
import itertools
import datetime
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


df_ile_d_fr['date']=date
df_ile_defrance=df_ile_d_fr[['date','Consommation']]
df_ile_defrance=df_ile_defrance.set_index('date')
df_ile_defrance=df_ile_defrance.fillna(method='bfill')
df_ile_defrance_heure=df_ile_defrance.resample('H').sum()
df_ile_defrance_mois=df_ile_defrance.resample('M').sum()

#fonction de recherche des paramètres oprtimaux

# On définit les paramètres p,d et entre 0 et 2
p = d = q = range(0, 3)

# Génère toutes les combinaisons différentes de triplets p, d et q
pdq = list(itertools.product(p, d, q))

# Génère toutes les combinaisons différentes de triplets saisonniers p, d et q
pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

### Recherche des grille de paramètres optimaux ###



# Définition de la fonction
def sarimax_gridsearch(ts, pdq, pdqs, maxiter=50, freq='M'):
    '''
    Entrée : 
        ts : vos données de séries temporelles
        pdq : combinaisons des ordres d' ARIMA
        pdqs : combinaisons des ordres saisonnières d' ARIMA à partir des données ci-dessus
        maxiter : nombre d'itérations
        frequency : par défaut='M' pour le mois. On peut modifier pour le mettre , 'D' pour le jour, 'H' pour l'heure, 'Y' pour l'année. 
        
    Retour :
        Imprime les 5 meilleures combinaisons de paramètres
        Renvoie le cadre de données des combinaisons de paramètres classées par BIC
    '''

    # Exécuter une recherche en grille avec les paramètres pdq et pdq saisonnier et obtenir la meilleure valeur BIC
    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts, 
                                                order=comb,
                                                seasonal_order=combs,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False,
                                                freq=freq)

                output = mod.fit(maxiter=maxiter) 
                ans.append([comb, combs, output.bic])
                print('SARIMAX {} x {}12 : BIC Calculated ={}'.format(comb, combs, output.bic))
            except:
                continue
            
    # Déterminer les paramètres avec une valeur BIC minimale

    # Convertir en dataframe
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'bic'])

    # Trier et renvoyer les 5 meilleures combinaisons
    ans_df = ans_df.sort_values(by=['bic'],ascending=True)[0:5]
    
    return ans_df


# séparations de données en données d'entrainement et de test
train_size=int(len(df_ile_defrance_mois) *0.7)
test_size=int(len(df_ile_defrance_mois))-train_size
X_train=df_ile_defrance_mois[:train_size]
X_test=df_ile_defrance_mois[train_size:]
#recherche des modèles optimaux selon le BIC
sarimax_gridsearch(X_train, pdq, pdqs, freq='M')

#Ajustant le premier modèle du grille de recherche SARIMA(1,2,2)(1,2,2,12)
sarima_1 = sm.tsa.statespace.SARIMAX(X_train, 
                                    order=(1,2,2), 
                                    seasonal_order=(1,2,2,12), 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False,
                                    freq='M')
                                    
# Ajustement du modèle
resultat_1= sarima_1.fit()
    
# resumé du resultat
print(resultat_1.summary())

#Tous les parametres ne sont pas significativement differents de zero
y_pred_1= resultat_1.predict(start =train_size, end=train_size+test_size-1)

#nous allons contruire une data avec les valeurs reélles et predites
Actuelle=pd.DataFrame(df_ile_defrance_mois.iloc[train_size:, 0])
y_pred_1=pd.DataFrame(y_pred_1)
y_pred_1.reset_index(drop=True, inplace=True)
y_pred_1.index=X_test.index
y_pred_1['valeur_réelle'] = Actuelle['Consommation']
y_pred_1.rename(columns={0:'valeur_predite'}, inplace=True)

#construction des intervalles de confiances
pred_int_conf=resultat_1.get_prediction(start=train_size, end=train_size+test_size-1)
int_conf=pred_int_conf.conf_int()


# calcule de metric de performance
## Calcule du MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_1=mean_absolute_percentage_error(y_pred_1['valeur_réelle'],y_pred_1['valeur_predite'])
print(mape_1)

#prevision à l'horizon 12
pred_1=resultat_1.predict(102, 113)#Prédiction

df_pred_1 = pd.concat([df_ile_defrance_mois, pred_1])#Concaténation des prédictions

#visiualisation des valeurs predites et actuelles
fig = plt.figure(figsize=(20,8))
plt.subplot(221)

plt.plot(y_pred_1['valeur_réelle'], color='blue', label='valeurs réelles')
plt.plot(y_pred_1['valeur_predite'], color='red', label='valeurs predite')
plt.fill_between(int_conf.index[1:],
                int_conf.iloc[1:, 0],
                int_conf.iloc[1:, 1], color='k', alpha=.2)
plt.ylabel('consommation')
plt.legend()
plt.title('Valeurs actuelles et prédites avec intervalle de confiance de 95%');
plt.subplot(222)

plt.plot(y_pred_1['valeur_réelle'], color='blue', label='valeurs réelles')
plt.plot(y_pred_1['valeur_predite'], color='red', label='valeurs predite')
plt.ylabel('consommation')
plt.legend()
plt.title('Valeurs actuelles et prédites');

plt.subplot(223)


plt.plot(df_pred_1) #Visualisation

plt.axvline(x= df_pred_1.index[102], color='red'); # Ajout de la ligne verticale

#Ajustant le premier modèle du grille de recherche SARIMA(2,2,2)(1,2,2,12)


sarimax2 = sm.tsa.statespace.SARIMAX(X_train, 
                                    order=(2,2,2), 
                                    seasonal_order=(1,2,2,12), 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False,
                                    freq='M')
                                    
# ajustement du modèle
resultat_2= sarimax2.fit()
    
# resumé des resultats
print(resultat_2.summary())

y_pred_2= resultat_2.predict(start =train_size, end=train_size+test_size-1)

pred_int_conf_2=resultat_2.get_prediction(start=train_size, end=train_size+test_size-1)

int_conf_2=pred_int_conf_2.conf_int()
#nous allons contruire une data avec les valeurs reélles et predites pôur le modèle 
y_pred_2=pd.DataFrame(y_pred_2)
y_pred_2.reset_index(drop=True, inplace=True)
y_pred_2.index=X_test.index
y_pred_2['valeur_réelle'] = Actuelle['Consommation']
y_pred_2.rename(columns={0:'valeur_predite'}, inplace=True)


#calcule de MAPE pour le modèle 3
mape_2=mean_absolute_percentage_error(y_pred_2['valeur_réelle'],y_pred_2['valeur_predite'])
print(mape_2)


#prevision à l'horizon 12
pred_2=resultat_2.predict(102, 113)#Prédiction

df_pred_2= pd.concat([df_ile_defrance_mois, pred_2])#Concaténation des prédictions

#visiualisation des valeurs predites et actuelles
fig = plt.figure(figsize=(20,8))
plt.subplot(221)

plt.plot(y_pred_2['valeur_réelle'], color='blue', label='valeurs réelles')
plt.plot(y_pred_2['valeur_predite'], color='red', label='valeurs predite')
plt.fill_between(int_conf_2.index[1:],
                int_conf_2.iloc[1:, 0],
                int_conf_2.iloc[1:, 1], color='k', alpha=.2)
plt.ylabel('consommation')
plt.legend()
plt.title('Valeurs actuelles et prédites avec intervalle de confiance de 95%');
plt.subplot(222)

plt.plot(y_pred_2['valeur_réelle'], color='blue', label='valeurs réelles')
plt.plot(y_pred_2['valeur_predite'], color='red', label='valeurs predite')
plt.ylabel('consommation')
plt.legend()
plt.title('Valeurs actuelles et prédites');

plt.subplot(223)


plt.plot(df_pred_2) #Visualisation

plt.axvline(x= df_pred_2.index[102], color='red'); # Ajout de la ligne verticale


## Identification des modèles par les autocorrélations simple et partielles

plt.plot(df_ile_defrance_mois);
# la série ne presente pas  une tendance mais par contre, elle a une saisonnalité de période 12

#Nous allons regarder l'autorrélation simple de la série pour voir la stationnarité 
pd.plotting.autocorrelation_plot(df_ile_defrance_mois);
# L'autocorélation simple tend vers zero mais presente des pic saisonnier.

#Nous allons différencier la série pour gommer saisonnalité et atteindre la stationnarité 
# nous effectuons une première différenciation simple

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) #Création de la figure et des axes

df_ile_defrance_mois_1=df_ile_defrance_mois.diff().dropna() #Différenciation d'ordre 1

df_ile_defrance_mois_1.plot(ax = ax1) #Série différenciée

pd.plotting.autocorrelation_plot(df_ile_defrance_mois_1, ax = ax2); #Autocorrélogramme de la série

#nous effectuons une deuxieme différenciation en saisonnalité

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

df_ile_defrance_mois_2=df_ile_defrance_mois_1.diff(periods=12).dropna() #Différenciation d'ordre 12

df_ile_defrance_mois_2.plot(ax = ax1) #Série doublement différenciée

pd.plotting.autocorrelation_plot(df_ile_defrance_mois_2, ax = ax2); #Autocorrélogramme de la série

# La serie semble  stationnaire essayon de confirme avec le test ADF
_, p_value, _, _, _, _  = sm.tsa.stattools.adfuller(df_ile_defrance_mois_2)
p_value  
#la p-valeur est inferieur à 5%, donc on considère la série comme étant stationnaire

# Comme nous avons stationnarisé la série nous allons modéliser avec le mod-èle SARIMA(p,d,q)(P,D,Q)_12
#pendnat le processus de stationnarité nous avons determiner les ordres d'inegrations.
# Ils nous reste à déterminer les ordres p,P,q et Q

#Visualiser sur 36 décalages les autocorrélogrammes simple et partiel de la série doublement différenciée

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

plot_acf(df_ile_defrance_mois_2, lags = 36, ax=ax1)
plot_pacf(df_ile_defrance_mois_2, lags = 36,method='ywmle', ax=ax2)
plt.show()

# vous allons commencer par essayer mododèle SARIMA(2,1,2)(1,1,1,12)

#Ajustant le modele SARIMA(2,1,2)(1,1,1,12)
model_manuelle_1 = sm.tsa.statespace.SARIMAX(X_train, 
                                    order=(2,1,2), 
                                    seasonal_order=(1,1,1,12), 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False,
                                    freq='M')
                                    
# Ajustement du modèle
resultat_manuelle_1=model_manuelle_1.fit()
    
# resumé du resultat
print(resultat_manuelle_1.summary())

# Tous les paramètres non saisonniers ont un p_value superieur 5% donc, on les éleminent du modèles
# Et nous avons le modèle suivant SARIMA(0,1,0)(1,1,1,12)
model_manuelle_2 = sm.tsa.statespace.SARIMAX(X_train, 
                                    order=(0,1,0), 
                                    seasonal_order=(1,1,1,12), 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False,
                                    freq='M')
                                    
# Ajustement du modèle
resultat_manuelle_2=model_manuelle_2.fit()
    
# resumé du resultat
print(resultat_manuelle_2.summary())

#Nous gardons ce modèle pour modéliser nos donnés 
y_pred_m2=resultat_manuelle_2.predict(start =train_size, end=train_size+test_size-1)

pred_int_conf_m2=resultat_manuelle_2.get_prediction(start=train_size, end=train_size+test_size-1)

int_conf_m2=pred_int_conf_m2.conf_int()

#nous allons contruire une data avec les valeurs reélles et predites
y_pred_m2=pd.DataFrame(y_pred_m2)
y_pred_m2.reset_index(drop=True, inplace=True)
y_pred_m2.index=X_test.index
y_pred_m2['valeur_réelle'] = Actuelle['Consommation']
y_pred_m2.rename(columns={0:'valeur_predite'}, inplace=True)

#calcule de MAPE pour le modèle 
mape_m2=mean_absolute_percentage_error(y_pred_m2['valeur_réelle'],y_pred_m2['valeur_predite'])
print(mape_m2)

#prevision à l'horizon 12
pred_m2 =resultat_manuelle_2.predict(102, 113)#Prédiction

df_pred_m2 = pd.concat([df_ile_defrance_mois, pred_m2])#Concaténation des prédictions

#visiualisation des valeurs predites et actuelles
fig = plt.figure(figsize=(20,8))
plt.subplot(221)

plt.plot(y_pred_m2['valeur_réelle'], color='blue', label='valeurs réelles')
plt.plot(y_pred_m2['valeur_predite'], color='red', label='valeurs predite')
plt.fill_between(int_conf_m2.index[1:],
                int_conf_m2.iloc[1:, 0],
                int_conf_m2.iloc[1:, 1], color='k', alpha=.2)
plt.ylabel('consommation')
plt.legend()
plt.title('Valeurs actuelles et prédites avec intervalle de confiance de 95%');
plt.subplot(222)

plt.plot(y_pred_m2['valeur_réelle'], color='blue', label='valeurs réelles')
plt.plot(y_pred_m2['valeur_predite'], color='red', label='valeurs predite')
plt.ylabel('consommation')
plt.legend()
plt.title('Valeurs actuelles et prédites');

plt.subplot(223)


plt.plot(df_pred_m2) #Visualisation

plt.axvline(x= df_pred_m2.index[102], color='red'); # Ajout de la ligne verticale

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
print(resultat_manuelle_6.summary())

y_pred_m6=resultat_manuelle_6.predict(start =train_size, end=train_size+test_size-1)

pred_int_conf_m6=resultat_manuelle_6.get_prediction(start=train_size, end=train_size+test_size-1)

int_conf_m6=pred_int_conf_m6.conf_int()

#nous allons contruire une data avec les valeurs reélles et predites
y_pred_m6=pd.DataFrame(y_pred_m6)
y_pred_m6.reset_index(drop=True, inplace=True)
y_pred_m6.index=X_test.index
y_pred_m6['valeur_réelle'] = Actuelle['Consommation']
y_pred_m6.rename(columns={0:'valeur_predite'}, inplace=True)

#calcule de MAPE pour le modèle 
mape_m6=mean_absolute_percentage_error(y_pred_m6['valeur_réelle'],y_pred_m6['valeur_predite'])
print(mape_m6)

pred_m6=resultat_manuelle_6.predict(102, 124)#Prédiction

df_pred_m6 = pd.concat([df_ile_defrance_mois, pred_m6])#Concaténation des prédictions

#visiualisation des valeurs predites et actuelles
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
plt.title('Valeurs actuelles et prédites');

plt.subplot(223)

plt.plot(df_pred_m6) #Visualisation

plt.title('previsions à horizon 12');
plt.axvline(x= df_pred_m6.index[len(X_train)+len(X_test)], color='red'); # Ajout de la ligne verticale

#### Utilisation du module Prophet. Prophet s'attend à ce que le format de données soit specifique.
#Le modèle attend une colonne "ds" qui contient le champ datetime et une colonne "y" qui contient les valeurs que
#nous allons modéliser. par consequent, nous allons renomer les colonnes de notre data

df_ile_defrance_pro_M=df_ile_defrance_mois.rename(columns={'Consommation':'y'})
df_ile_defrance_pro_M=df_ile_defrance_pro_M.reset_index('date')
df_ile_defrance_pro_M=df_ile_defrance_pro_M.rename(columns={'date':'ds'})
train_size_p=int(len(df_ile_defrance_pro_M) *0.7)
test_size_p=int(len(df_ile_defrance_pro_M))-train_size_p
X_train_pro=df_ile_defrance_pro_M[:train_size_p]
X_test_pro=df_ile_defrance_pro_M[train_size_p:]
#modélisation avec prophet
model_prophet = Prophet()
model_prophet.fit(X_train_pro)
#Le module permet de initialiser une dataframe qui vide qui contiendra tous les sorties des predictions
#Le modele utlise l'historique de tous les données pour faire de prevision
future = model_prophet.make_future_dataframe(periods=31, freq='M')
futur_forecast = model_prophet.predict(df=future)
#y_pred_prophet contiendra les valeurs predites, les valeurs actuelles et
#les intervalles de confiances de valeurs predites

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
#calcule de mape
mape_pro1=mean_absolute_percentage_error(y_pred_prophet['valeur_réelle'],y_pred_prophet['valeur_predite'])
mape_pro1
#visiualisation des valeurs predites et actuelles

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
plt.legend() ; 

#Nous allons ajuster un modèle qui ne prend en compte l'historique des données.
future_sh = model_prophet.make_future_dataframe(periods=31, freq='M',include_history = False)
futur_forecast_sh = model_prophet.predict(df=future_sh)

#Nous construissons un dataframe y_pred_prophet_sh
liste=['yhat_lower','yhat_upper','yhat']
y_pred_prophet_sh=futur_forecast_sh[liste]

y_pred_prophet_sh=pd.DataFrame(y_pred_prophet_sh)
y_pred_prophet_sh=y_pred_prophet_sh.rename(columns={'yhat':'valeur_predite'})
y_pred_prophet_sh.reset_index(drop=True, inplace=True)
y_pred_prophet_sh.index=X_test_pro.index
y_pred_prophet_sh['ds']=X_test_pro.ds
y_pred_prophet_sh['valeur_réelle']=X_test_pro.y
y_pred_prophet_sh=y_pred_prophet_sh.rename(columns={'ds':'date'})
y_pred_prophet_sh=y_pred_prophet_sh.set_index('date')
#metric de performance
mape_pro_sh=mean_absolute_percentage_error(y_pred_prophet_sh['valeur_réelle'],y_pred_prophet_sh['valeur_predite'])
mape_pro_sh

#Nous constatons le modèle est moins performant quand il ne prend pas en compte le données historiques

#prediction à l'horizon 24
future_horiz_24 = model_prophet.make_future_dataframe(periods=53, freq='M',include_history = False)
futur_forecast_horiz_24 = model_prophet.predict(df=future_horiz_24)

y_pred_horiz_n=futur_forecast_horiz_24[['ds','yhat']]
y_pred_horiz_n=pd.DataFrame(y_pred_horiz_n)
y_pred_horiz_n=y_pred_horiz_n.rename(columns={'yhat':'valeur_predite'})
y_pred_horiz_n=y_pred_horiz_n.rename(columns={'ds':'date'})
y_pred_horiz_n=y_pred_horiz_n.set_index('date')
y_pred_horiz_24=y_pred_horiz_n[len(X_test):]
df_pred_horiz_24= pd.concat([df_ile_defrance_mois,y_pred_horiz_24])#Concaténation des prédictions

#visiualisation des valeurs predites et actuelles

fig = plt.figure(figsize=(20,10))
plt.subplot(221)
plt.plot(y_pred_prophet_sh['valeur_predite'], label='valeurs prédites')
plt.plot(y_pred_prophet_sh['valeur_réelle'], label='valeurs actuelles')
plt.xlabel('date')
plt.ylabel('consommation')
plt.title(' Valeurs prédites et actuelles')
plt.legend() ; 
plt.subplot(222)
plt.plot(y_pred_prophet_sh['valeur_predite'], label='valeurs prédites')
plt.plot(y_pred_prophet_sh['valeur_réelle'], label='valeurs actuelles')

plt.fill_between(y_pred_prophet_sh['yhat_lower'].index,
                y_pred_prophet_sh['yhat_upper'],
                y_pred_prophet_sh['yhat_lower'], color='b', alpha=.2)
#plt.plot(y_pred_prophet['yhat_upper'],'--', label='valeurs prédites maximum')
#plt.plot(y_pred_prophet['yhat_lower'],'--',label='valeurs prédites minimum');
plt.xlabel('date')
plt.ylabel('consommation')
plt.title(' Valeurs prédites et actuelles avec intervalle de confiance')
plt.legend() ; 
plt.subplot(223)


plt.plot(df_pred_horiz_24) #Visualisation
plt.xlabel("Année")
plt.ylabel("Consommation")
plt.title("previsions à l'horizon 24");
plt.axvline(x= df_pred_m6.index[len(X_train)+len(X_test)], color='red'); # Ajout de la ligne verticale