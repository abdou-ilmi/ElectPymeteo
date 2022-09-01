#DATAVISUALISATION A PARTIR DES DONNEES A L'ECHELLE NATIONALE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

df0 = pd.read_csv("C:\\DataScientest\\eco2mix-national-cons-def.csv", sep =";")

#Premier preprocessing : nettoyage du dataframe
def preprocessing_nettoyage(df):
    df = df.replace('ND', np.nan)
    df = df[df["Consommation (MW)"].notna()]
    df[["Ech. comm. Allemagne-Belgique (MW)","Gaz - Cogénération (MW)"]] = \
    df[["Ech. comm. Allemagne-Belgique (MW)","Gaz - Cogénération (MW)"]].astype(float)
    #df = df.dropna(axis=1)  #cela supprime la colonne "Gaz - Cogénération (MW)" car pas de données en 2012
    df = df.drop(["Prévision J-1 (MW)","Prévision J (MW)","Périmètre","Nature","Heure"], axis = 1)
    return df

df1 = df0.copy()
df1 = preprocessing_nettoyage(df1)


#Recherche de corrélations à l'aide d'une HeatMap
fig, ax = plt.subplots(figsize=(22,20))
sns.heatmap(df1.corr(), annot=True, ax=ax, cmap = 'coolwarm')
plt.show()


#Visualisation de la corrélation entre l'électricité produite par les énergies fossiles et le taux de CO2 du mix électrique
plt.figure(figsize=(15,10))
plt.scatter(df1.groupby("Date").mean()['Charbon (MW)']+df1.groupby("Date").mean()['Fioul (MW)']+
            df1.groupby("Date").mean()['Gaz (MW)'],df1.groupby("Date").mean()['Taux de CO2 (g/kWh)'],color='red')
plt.ylabel('Taux de CO2 (g/kWh)')
plt.xlabel("Production électrique par combustion d'hydocarbures (MW)");
plt.show()


#Deuxième préprocessing : on conserve uniquement la consommation et les productions par filières et on formate correctement
#la colonne temporelle
def preprocessing_selection(df):
    df = df.iloc[:,1:12]
    df['Date et Heure'] = df['Date et Heure'].apply(lambda x: x[:-6])
    df['Date et Heure'] = pd.to_datetime(df['Date et Heure'], format = "%Y-%m-%dT%H:%M:%S")
    df = df.sort_values(by = ["Date et Heure"])
    df = df.set_index('Date et Heure')
    return df

df = df1.copy()
df = preprocessing_selection(df)


#Graphiques permettant de visualiser l'importance des filières de production les unes par rapport aux autres et l'évolution
#de la consommation et de la production par filière entre 2012 et 2021
fig1 = plt.figure(figsize=[15,20])
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

ax1.boxplot(df.loc[:,'Fioul (MW)' : 'Bioénergies (MW)'].resample("W").mean(), showfliers=False)
ax1.set_xticklabels(['Fioul','Charbon','Gaz','Nucléaire','Eolien','Solaire','Hydraulique','Pompage','Bioénergies'])
ax1.set_ylabel("Production (MW)")
ax1.set_title("Boites à moustache des moyennes hebdomadaires de puissance de production (2012 à 2021)");

liste_graphes = ['Consommation (MW)', 'Gaz (MW)', 'Nucléaire (MW)','Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)']
ax2.plot(df.loc[:,liste_graphes].resample("W").mean(), label = liste_graphes)
ax2.legend()
ax2.set_xlabel("Temps")
ax2.set_ylabel("Puissance (MW)")
ax2.set_title("Consommation et productions lissées par semaine (2012 à 2021)")
plt.show()


#Graphiques permettant de visualiser l'apparition de motifs élémentaires dans la consommation et certaines filières de 
#production suivant l'échelle de temps considérée
fig2 = plt.figure(figsize=[20,20])
ax21 = fig2.add_subplot(221)
ax22 = fig2.add_subplot(222,sharey=ax21)
ax23 = fig2.add_subplot(223,sharey=ax21)
ax24 = fig2.add_subplot(224,sharey=ax21)

ax21.plot(df.loc["2019",liste_graphes].resample("D").mean(), label = liste_graphes)
ax21.legend()
ax21.set_xlabel("Temps")
ax21.set_ylabel("Puissance (MW)")
ax21.set_title("Consommation et productions lissées par jour (2019)")

ax22.plot(df.loc["2019-06",liste_graphes])
ax22.legend()
ax22.set_xlabel("Temps")
ax22.set_title("Consommation et productions (Juin 2019)")

ax23.plot(df.loc["2019-06-03" : "2019-06-09",liste_graphes])
ax23.legend()
ax23.set_xlabel("Temps")
ax23.set_ylabel("Puissance (MW)")
ax23.set_title("Consommation et productions (semaine du 3 au 9 juin 2019)")

ax24.plot(df.loc["2019-06-04",liste_graphes])
ax24.legend()
ax24.set_xlabel("Temps")
ax24.set_title("Consommation et productions (4 juin 2019)")
plt.show()


#Boîtes à moustache permettant de confirmer certaines saisonnalités précédemment observées
fig3 = plt.figure(figsize=[20,16])
ax31 = fig3.add_subplot(221)
ax32 = fig3.add_subplot(222)
ax33 = fig3.add_subplot(223)
ax34 = fig3.add_subplot(224)

l=list()
for i in df.resample("M").mean().index.month.unique():
    l.append(df.resample("M").mean()[df.resample("M").mean().index.month == i]['Consommation (MW)'])
ax31.boxplot(l, showfliers=False)
plt.sca(ax31)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['J', 'F', 'M', 'A','M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax31.set_ylabel("Consommation (MW)")

l=list()
for i in df.resample("M").mean().index.month.unique():
    l.append(df.resample("M").mean()[df.resample("M").mean().index.month == i]['Nucléaire (MW)'])
ax32.boxplot(l, showfliers=False)
plt.sca(ax32)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['J', 'F', 'M', 'A','M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax32.set_ylabel("Nucléaire (MW)")

l=list()
for i in df.resample("M").mean().index.month.unique():
    l.append(df.resample("M").mean()[df.resample("M").mean().index.month == i]['Eolien (MW)'])
ax33.boxplot(l, showfliers=False)
plt.sca(ax33)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['J', 'F', 'M', 'A','M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax33.set_ylabel("Eolien (MW)")

l=list()
for i in df.resample("M").mean().index.month.unique():
    l.append(df.resample("M").mean()[df.resample("M").mean().index.month == i]['Solaire (MW)'])
ax34.boxplot(l, showfliers=False)
plt.sca(ax34)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['J', 'F', 'M', 'A','M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax34.set_ylabel("Solaire (MW)")
plt.show()

##############################################################################################################
##############################################################################################################
##############################################################################################################

#DATAVISUALISATION A PARTIR DES DONNEES A L'ECHELLE REGIONALE 

#preparation de donnée
df=pd.read_csv('donnee_regional.csv', sep=';', parse_dates=[3], squeeze=True, usecols=[0,1,2,3,4,5,6])
df=df.rename(columns={'Consommation (MW)':'Consommation'})
#supression des colonnes non utilisées
df=df.drop(['Code INSEE région','Nature','Date','Heure','Date - Heure'],axis=1)
#création des sous data selon certaines régions

df_Bour_FC=df[(df.Région=='Bourgogne-Franche-Comté')]
df_ile_d_fr=df[(df.Région=='Île-de-France')]
#variabe datatime d'indexation
date= pd.date_range("2013-01-01", periods=148944, freq="0.5H")

#donnée de la région de bourgone-Franche-comté
df_Bour_FC['date']=date
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

#Somme de la consommation journalier

som_de_conso_par_jour=df_Bour_FC_conso.groupby('day_name').sum().Consommation.reset_index()
#representation de la consommation en fonction des jours pour avoir
#une idée sur le jour où la consommation est la plus élévée
plt.figure(figsize=(8, 6))
order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sns.barplot(x=som_de_conso_par_jour['day_name'], y=som_de_conso_par_jour['Consommation'], order=order)
plt.xlabel('jour')
plt.title('consommation par rapport aux jours')
plt.show;
#Nous constatons sur cette période, la consommation cumulée sur les jours
#est plus importante le mercredi et le jeudi
#Somme de la consommation mensuelle

som_de_conso_par_mois=df_Bour_FC_conso.groupby('month_name').sum().Consommation.reset_index()
#representation de la consommation en fonction des mois pour avoir
#une idée sur le mois où la consommation est la plus élévée
plt.figure(figsize=(10, 6))
order_m=['January','February','March','April','May','June','July','September','October','November','December']
sns.barplot(x=som_de_conso_par_mois['month_name'], y=som_de_conso_par_mois['Consommation'], order=order_m)
plt.xlabel('mois')
plt.title('consommation par rapport aux mois')
plt.show;
#Nous constatons sur cette période, la consommation cumulée sur les mois
#est plus importante en janvier.

#Somme de la consommation mensuelle

som_de_conso_par_heure=df_Bour_FC_conso_b.groupby('hour').sum().Consommation.reset_index()
#representation de la consommation en fonction des heures pour avoir
#une idée sur l'heure où la consommation est la plus élévée
plt.figure(figsize=(10, 6))
#order_m=['January','February','March','April','May','June','July','September','October','November','December']
sns.barplot(x=som_de_conso_par_heure['hour'], y=som_de_conso_par_heure['Consommation'])
plt.xlabel('heure')
plt.title('consommation par rapport aux heures')
plt.show;
#Nous constatons sur cette période, la consommation cumulée sur les heures
#est plus importante à 12, 11.

#Consommation d'une semaine type de chaque saison en 2013 et 2019
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
# Nous observons que le profile de consommation est similaire en 2013 et en 2019, donc absence de tendance.
# Par contre nous avons une periodité marqué en fonction de la saison. 

#Consommation d'un mois type de chaque saison en 2013 et 2019

plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2013-01-01 00:00:00':'2013-01-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2013-01-01 00:00:00':'2013-01-30 23:00:00'],
         label='hiver')
plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2013-04-01 00:00:00':'2013-04-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2013-04-01 00:00:00':'2013-04-30 23:00:00'],
         label='Printemps')
plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2013-07-01 00:00:00':'2013-07-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2013-07-01 00:00:00':'2013-07-30 23:00:00'],
         label='été')

plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2013-10-01 00:00:00':'2013-10-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2013-10-01 00:00:00':'2013-10-30 23:00:00'],
         label='automne')
#plt.plot(range(len(df_est1_journee_F20)),df_est1_journee_F20['Consommation (MW)'], label='4 Février 2020')
#plt.title('Consommation de la Region de Grand Est pour un jour donné')
plt.xlabel('jour')
plt.title('Consommation de 2013')
plt.legend();
plt.subplot(1,2,2)


plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2019-01-01 00:00:00':'2019-01-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2019-01-01 00:00:00':'2019-01-30 23:00:00'],
         label='hiver')
plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2019-04-01 00:00:00':'2019-04-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2013-04-01 00:00:00':'2013-04-30 23:00:00'],
         label='Printemps')
plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2019-07-01 00:00:00':'2019-07-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2019-07-01 00:00:00':'2019-07-30 23:00:00'],
         label='été')

plt.plot(range(len(df_Bour_FC_conso_jour['Consommation'].loc['2019-10-01 00:00:00':'2019-10-30 23:00:00']))
         ,df_Bour_FC_conso_jour['Consommation'].loc['2019-10-01 00:00:00':'2019-10-30 23:00:00'],
         label='automne')
#plt.plot(range(len(df_est1_journee_F20)),df_est1_journee_F20['Consommation (MW)'], label='4 Février 2020')
#plt.title('Consommation de la Region de Grand Est pour un jour donné')
plt.xlabel('jour')
plt.title('Consommation de 2019')
plt.legend()
plt.suptitle("Consommation d'un mois type de chaque saison en 2013 et 2019",fontsize=20);

##############################################################################################################
##############################################################################################################
##############################################################################################################

#DATAVISUALISATION DE L'IMPLANTATION DES EOLIENNES EN PAYS DE LA LOIRE

import pandas as pd

df = pd.read_csv("234400034_eolien-terrestre-mats-deoliennes-en-pays-de-la-loire0.csv", sep = ";")

def preprocessing(df):
    df = df[df["en_service"]=="OUI"]
    df["latitude"] = df["geo_point_2d"].apply(lambda x : x.split(",")[0])
    df["longitude"] = df["geo_point_2d"].apply(lambda x : x.split(",")[1])
    return df[["latitude","longitude","pu_nominal","date_prod"]]

df0 = df.copy()
df = df0.copy()
df = preprocessing(df)

nantes = [47.15, -1.61, "Nantes"]
alencon = [48.45, 0.11, "Alençon"]
tours = [47.44, 0.73, "Tours"]
la_rochelle = [46.04, -1.405, "La Rochelle"]

#############################################################################

#CREATION DE NOUVELLES COLONNES AVEC LA STATION METEO LA PLUS PROCHE ET LA DISTANCE CORRESPONDANTE
import numpy as np
from math import sin, cos, acos, pi
#############################################################################
def deg2rad(dd):
    """Convertit un angle "degrés décimaux" en "radians"
    """
    return dd/180*pi
#############################################################################
def distanceGPS(ville):
    latA = deg2rad(latitude)
    longA = deg2rad(longitude)
    latB = deg2rad(ville[0])
    longB = deg2rad(ville[1])
    
    """Retourne la distance en mètres entre les 2 points A et B connus grâce à
       leurs coordonnées GPS (en radians).
    """
    # Rayon de la terre en mètres (sphère IAG-GRS80)
    RT = 6378137
    # angle en radians entre les 2 points
    S = acos(sin(latA)*sin(latB) + cos(latA)*cos(latB)*cos(abs(longB-longA)))
    # distance entre les 2 points, comptée sur un arc de grand cercle
    return round(S*RT)
#############################################################################
dico_station = {"Nantes" : 7222, "Alençon" : 7139, "Tours" : 7240, "La Rochelle" : 7314}
#############################################################################
df["latitude"] = df["latitude"].astype(float)
df["longitude"] = df["longitude"].astype(float)
dist_min = [0]*df.shape[0]
station_proche = []
for i in range(df.shape[0]):
    latitude = df.iloc[i,0]
    longitude = df.iloc[i,1]
    dist_nantes = distanceGPS(nantes)
    dist_alencon = distanceGPS(alencon)
    dist_tours = distanceGPS(tours)
    dist_la_rochelle = distanceGPS(la_rochelle)
    dist_min[i] = min(dist_nantes, dist_alencon, dist_tours, dist_la_rochelle)
    if dist_min[i] == dist_nantes :
        station_proche.append(nantes[2])
    elif dist_min[i] == dist_alencon :
        station_proche.append(alencon[2])
    elif dist_min[i] == dist_tours :
        station_proche.append(tours[2])
    else :
        station_proche.append(la_rochelle[2])

df["Station la plus proche"] = station_proche
df["Distance à la station (m)"] = dist_min
df["Numéro de station"] = df["Station la plus proche"].apply(lambda x : dico_station[x])

#############################################################################

#CREATION DE LA CARTE
import folium
import json
from folium.plugins import BeautifyIcon

lat_centre = 47.4  
long_centre = -0.9   
zoom = 8
carte= folium.Map(location=[lat_centre,long_centre],zoom_start=zoom)

for i in range(df.shape[0]):
    latitude = df.iloc[i,0]
    longitude = df.iloc[i,1]
    puissance = df.iloc[i,2]
    m = "<strong>" + "Puissance : " + "</strong>" + str(puissance) + " MW"\
    + "<br>Station la plus proche : " + df.iloc[i,4]
    folium.CircleMarker([latitude, longitude], radius = 5,color=None,fill_color ="red",
                    fill_opacity=0.5,popup = folium.Popup(m, max_width = 400)).add_to(carte)

folium.Marker([nantes[0],nantes[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = nantes[2]).add_to(carte)
folium.Marker([alencon[0], alencon[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = alencon[2]).add_to(carte)
folium.Marker([tours[0], tours[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = tours[2]).add_to(carte)
folium.Marker([la_rochelle[0], la_rochelle[1]], icon=folium.Icon(color='blue', icon='cloud'),popup = la_rochelle[2]).add_to(carte)

carte.save("Carte_éoliennes.html")