'''
The suggested algorithm written in Czech follows. 


Navrhovaný postup

Výpočet vždy v plovoucím časovém okně o pevné délce (W = 5 let?) s posuvem třeba po měsíci nebo po roce (dále dW). Mezi sousedními okny tedy bude vždy značný překryv. V každém okně se vždy pro referenční datovou řadu znovu spočítají převodní koeficienty přes ostatní pozorovatele. 

První okno musí vyjít z reálného pozorovatele, navrhuje se Zloch někde uprostřed své kariéry. Pak se okna posouvají tam i zpět v čase. 

V rámci okna se vždy nejprve spočítají převodní koeficienty Franty na ostatní dostupné vhodné (s více než 100 pozorováními?) pozorovatele. Tyto koeficienty se použijí pro "předpověď" na následný interval dW mimo okno. Jako optimalizovaná řada se uloží dW pozorování ze začátku okna. Okno W se posune od dW v čase a postup se opakuje. Po každém kroku v řadě přibyde dW interval vypočtených hodnot, do dalšího kroku vstupuje předpověď dW hodnot "před" aktuálním oknem. V následném okně se vždy upraví koeficienty a předpověď tak bude s novými. 

Pro doplnění minulých pozorování lze posouvat okno v čase zpět. Tvorba řady je pak z druhé strany. Je třeba zajistit kontinuitu. 

Problémy?
    
* Jak zjistit, jestli výpočet "neujíždí" někam? (Nápad: tentýž postup zpět v čase -- sejdeme se?) Nebo kontrola k_i pro stejné dvojice v různých W -- sekulární trend pro více k je problém. 
* Předpověď bude vždy pro Q=5. Pak se budou blbě hledat dvojice pro výpočet k. Nešlo by tedy pozorování Franty redukovat na Q pozorovatele? Lze výhybkou: pokud jsou k dispozici dvojice datum,Q, vzít pro koeficienty ty, pak se podívat po Q_Franta=5 a k nim dohledat jiná pozorování při témže dni, g a f pro Frantu pak redukovat z Q=5 na jiné Q odpovídající druhému pozorovateli. 
* Interně držet g a f jako desetinné číslo, na celé převádět až úplně na konci. 

'''


import pandas as pd 
import datetime as dt
from os import listdir
from os.path import isfile, join, isdir
import code
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import copy
from prettytable import PrettyTable

####################################
# PARAMETERS
#

# window widths
W=dt.timedelta(days=5000)  # width of the evaluation window
dW=dt.timedelta(days=100)   # width of the prediction window

# dealing with low g and low f values that are not to be considered in the calculation 
minspots=5   # when doing the k table, how many spots there have to be at minimum?
mingroups=1  # when doing the k table, how many groups there have to be at minimum?

# reduction to the optimal conditions
# R, f, g
#redukce=np.array([-0.07877999, -0.16307175, -0.04876931])    # fit with no uncerstainties
redukce=np.array([-0.09048021, -0.07796618, -0.03477956])   # fit with errors in Q +/- 0.5

# turning on/off reductions
redukce1=1  # reduce during the evaluation period, default=1
redukce2=0 # reduce during the prediction period, default=0

Npruchodu=2 # how many passes

################################################################################################

# read in the database file

db = pd.read_hdf('databaze.hdf','db')

# declare usefull functions. Modify Numpy functions so that they return NaN in the case all the
# elements of the tested array are NaN. Numpy returns an empty array in that case, which crashes 
# the program

def nansum(ar):
    if np.sum(np.isnan(ar)) == len(ar):
        return np.NaN
    else:
        return np.nansum(ar)

def nanstd(ar):
    if np.sum(np.isnan(ar)) == len(ar):
        return np.NaN
    else:
        return np.nanstd(ar)

# prepare the list of available observers:
# HERE
# pozorovatele, kteri jsou k dispozici, v "testovani" jsou jejich unikatni jmena
pozorovatele=db['Pozorovatel']
testovani=list(pozorovatele.drop_duplicates())
print(len(testovani))

# vyber pozorovatelu, kteri maji alespon 100 pozorovani, ti jsou pak v dlouhodobi
dlouhodobi=copy.deepcopy(testovani)
for pozorovatel in testovani:
    # print(pozorovatel)
    print(pozorovatel, ' ', len(db[db['Pozorovatel']==pozorovatel]))
    if len(db[db['Pozorovatel']==pozorovatel]) < 100:
        dlouhodobi.remove(pozorovatel)
        
# urezat jine nez dlouhodobe pozorovatele
db=db[db['Pozorovatel'].apply((lambda x: x in dlouhodobi) )]


# pozorovatele, kteri jsou k dispozici

referencni=copy.deepcopy(testovani)
#referencni=copy.deepcopy(dlouhodobi)

# referencni.remove(kralik)

table_k=pd.DataFrame({'Datum':[], 'Jmeno':[], 'k_f':[], 'k_g':[], 'Pruchod':[]})


##############################
# INITIAL MODEL

###
# 1. a reference observer

vzorny_pozorovatel='ZLOCH FRANTIŠEK'
from sunspots import initial_reference
predikce=initial_reference('databaze.hdf', vzorny_pozorovatel)
# from sunspots import initial_flat
# predikce=initial_flat('databaze.hdf', vzorny_pozorovatel)
kralik=vzorny_pozorovatel

#####################
# inicialni datova series. Bud jako prosty prumer vsech nebo jako jeden z vybranych pozorovatelu


# from sunspots import initial_mean
# predikce=initial_mean('databaze.hdf')
# kralik='PRUMER'


initial=copy.deepcopy(predikce)  # POZOR, s timto polem se dale nikde nepocita, az na konci, pro ucely porovnani puvodni a vysledne serie

startdat=min(predikce['Datum'])

# Uriznout inicialni sadu jen na prvni interval, pak uz se bude doplnovat predpovedi
dat1=startdat
dat2=startdat+W
predikce=predikce[(predikce['Datum'] >= dat1) & (predikce['Datum'] < dat2)]

#######################################

# Hlavni cyklus cesta tam

#######################################


##########
# pokus o vypocet nejistot 
# nejistota funkce f(x_1, x_2, ...) s velicinami x_1, ... a jejich nejistotami sigma_x1, ... se vypocte jako
# sigma_f = sqrt{ suma_i [ (\partial f / \partial x_i)^2 sigma_xi^2 ] }
# takze pro R = 10*g + f
# sigma_R = sqrt{ 10^2*sigma_g^2 + sigma_f^2 }
#
# nejprve zkopirovat puvodni pole kvuli konzistenci
nejistota=copy.deepcopy(predikce)
# pak vynulovat
nejistota['g']=0
nejistota['f']=0
nejistota['R']=0
# a vymazat sloupec Q, ktery je k nicemu
nejistota=nejistota.drop(columns=['Q'])

# import code
# code.interact(local=locals())

for pruchod in range(1,Npruchodu+1):
    #referencni=copy.deepcopy(testovani)
    # if pruchod == 1:
        # referencni.remove(kralik)
    # referencni.remove(kralik)
    print('*********************')
    print('Pruchod c. '+str(pruchod))
    print('Tam...')
    while (startdat+W+dW) < max(db['Datum']):
        dat1=startdat
        print(dat1, max(db['Datum']))
        dat2=startdat+W
        # uriznout databazi jen na zvoleny casovy usek
        db1=db[(db['Datum'] >= dat1) & (db['Datum'] < dat2)]


        # prepocitat koeficienty pro predikci
        tablekf=pd.DataFrame(columns=[kralik], index=referencni)
        tablekg=pd.DataFrame(columns=[kralik], index=referencni)
        tablenf=pd.DataFrame(columns=[kralik], index=referencni)
        tableng=pd.DataFrame(columns=[kralik], index=referencni)
        jmeno1=kralik
        # set1=db1[(db1['Pozorovatel'] == jmeno1)]
        set1=predikce[(predikce['Datum'] >= dat1) & (predikce['Datum'] < dat2)]
        for jmeno2 in referencni:
            set2=db1[(db1['Pozorovatel'] == jmeno2)]   
            # slouceni obou datasetu s prunikem po polich "on"  
            temp = pd.merge(set1, set2, how='inner', on=['Datum'])  # slucujeme po poli datum
            Q1=np.asarray(temp['Q_x'])
            Q2=np.asarray(temp['Q_y'])
            f1=np.asarray(temp['f_x'])
            f2=np.asarray(temp['f_y'])
            f1red=f1*(1-redukce1*redukce[1]*(5-Q1))
            f2red=f2*(1-redukce1*redukce[1]*(5-Q2))
            mask = (f1 > minspots) & (f2 > minspots)
            junk=f1red[mask]/f2red[mask]    
            k=np.mean(junk[np.isfinite(junk)])
            kf=k
            tablekf[jmeno1][jmeno2]=k
            tablenf[jmeno1][jmeno2]=len(np.isfinite(junk))
            g1=np.asarray(temp['g_x'])
            g2=np.asarray(temp['g_y'])
            g1red=g1*(1-redukce1*redukce[2]*(5-Q1))
            g2red=g2*(1-redukce1*redukce[2]*(5-Q2))
            mask = (g1 > mingroups) & (g2 > mingroups)
            junk=g1red[mask]/g2red[mask]
            k=np.mean(junk[np.isfinite(junk)])
            tablekg[jmeno1][jmeno2]=k
            tableng[jmeno1][jmeno2]=len(np.isfinite(junk))
            table_k=pd.concat([table_k, pd.DataFrame({'Datum':[startdat+W/2], 'Jmeno':[jmeno2], 'k_f':[kf], 'k_g':[k], 'Pruchod':[pruchod-0.5]})])
        # code.interact(local=locals())

        # posouvame se za hranici intervalu
        dat1=startdat+W
        dat2=startdat+W+dW
        # uriznout databazi jen na zvoleny casovy usek
        db2=db[(db['Datum'] >= dat1) & (db['Datum'] < dat2)]
        # najit pozorovatele, kteri se daji pouzit
        pozorovatele=db2['Pozorovatel']
        temp=list(pozorovatele.drop_duplicates())
        dostupni=copy.deepcopy(temp)
        dostupni=[jmeno for jmeno in dostupni if jmeno in referencni] # pouzit jen takove pozorovatele, kteri jsou soucasne i mezi duveryhodnymi
        #import code
        #code.interact(local=locals())
        # if (kralik in dostupni):
       #      dostupni.remove(kralik)
        #if (kralik in dostupni) & (pruchod == 1):
        #    dostupni.remove(kralik)
        print(dostupni)    
        if len(dostupni) == 0:
            break


        # code.interact(local=locals())
        for irow in range(0,len(db2)):
            date=db2.iloc[irow]['Datum']
            g=np.array(range(0,len(dostupni)))*np.nan
            f=np.array(range(0,len(dostupni)))*np.nan
            g_red=np.array(range(0,len(dostupni)))*np.nan
            f_red=np.array(range(0,len(dostupni)))*np.nan
            wg=np.array(range(0,len(dostupni)))*np.nan    
            wf=np.array(range(0,len(dostupni)))*np.nan 
            for iref in range(0,len(dostupni)):
                ref=dostupni[iref]
                if len(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['g']):
                    # pokud existuje porovnavaci udaj, tak nastav patricne hodnoty g, f a vah. Prepocitej g a f pres tabulku
                    g[iref]=np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['g'])*tablekg[kralik][ref]
                    f[iref]=np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['f'])*tablekf[kralik][ref]
                    g_red[iref]=g[iref]*(1-redukce2*redukce[2]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q'])))
                    f_red[iref]=f[iref]*(1-redukce2*redukce[1]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q'])))
                    # print((1-redukce[2]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q']))))
                    # print((1-redukce[1]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q']))))
                    wg[iref]=tableng[kralik][ref]
                    wf[iref]=tablenf[kralik][ref]
            if pruchod > 1:
                gsave=(nansum(g_red*(wg/nansum(wg))))
                fsave=(nansum(f_red*(wf/nansum(wf))))
                sigma_g=nanstd(g_red)
                sigma_f=nanstd(f_red)
            else: 
                gsave=(nansum(g*(wg/nansum(wg))))
                fsave=(nansum(f*(wf/nansum(wf))))
                sigma_g=nanstd(g)
                sigma_f=nanstd(f)
            # if np.isfinite(gsave):
                # import code
                # code.interact(local=locals())
            # gsave=(nansum(g*(wg/nansum(wg))))
            # fsave=(nansum(f*(wf/nansum(wf))))
            if np.isfinite(gsave):
                gsave=round(gsave)
            if np.isfinite(fsave):
                fsave=round(fsave)
            Rsave=10*gsave+fsave
            sigma_R=np.sqrt(10**2*sigma_g**2+sigma_f**2)
            if np.isfinite(Rsave):
                predikce=pd.concat([predikce, pd.DataFrame({'Datum':date, 'g':[gsave], 'f':[fsave], 'R':[Rsave], 'Q':[5]})])
                nejistota=pd.concat([nejistota, pd.DataFrame({'Datum':date, 'g':[sigma_g], 'f':[sigma_f], 'R':[sigma_R]})])

        startdat=startdat+dW

    ###############################################################################

    # code.interact(local=locals())

    #######################################

    # Hlavni cyklus cesta zpet

    #######################################
    print('Zpet...')
    startdat=max(predikce['Datum'])
    predikce_zpet=predikce[(predikce['Datum'] <= startdat) & (predikce['Datum'] > (startdat-W))]
    nejistota_zpet=nejistota[(nejistota['Datum'] <= startdat) & (nejistota['Datum'] > (startdat-W))]
    while (startdat-W-dW) > min(db['Datum']):
        dat1=startdat-W
        print(dat1, min(db['Datum']))
        dat2=startdat
        # uriznout databazi jen na zvoleny casovy usek
        db1=db[(db['Datum'] > dat1) & (db['Datum'] <= dat2)]


        # prepocitat koeficienty pro predikci
        tablekf=pd.DataFrame(columns=[kralik], index=referencni)
        tablekg=pd.DataFrame(columns=[kralik], index=referencni)
        tablenf=pd.DataFrame(columns=[kralik], index=referencni)
        tableng=pd.DataFrame(columns=[kralik], index=referencni)
        jmeno1=kralik
        # set1=db1[(db1['Pozorovatel'] == jmeno1)]
        set1=predikce_zpet[(predikce_zpet['Datum'] > dat1) & (predikce_zpet['Datum'] <= dat2)]
        for jmeno2 in referencni:
            set2=db1[(db1['Pozorovatel'] == jmeno2)]   
            # slouceni obou datasetu s prunikem po polich "on"  
            temp = pd.merge(set1, set2, how='inner', on=['Datum'])  # slucujeme po poli datum
            Q1=np.asarray(temp['Q_x'])
            Q2=np.asarray(temp['Q_y'])
            f1=np.asarray(temp['f_x'])
            f2=np.asarray(temp['f_y'])
            f1red=f1*(1-redukce1*redukce[1]*(5-Q1))
            f2red=f2*(1-redukce1*redukce[1]*(5-Q2))
            mask = (f1 > minspots) & (f2 > minspots)
            junk=f1red[mask]/f2red[mask]    
            k=np.mean(junk[np.isfinite(junk)])
            kf=k
            tablekf[jmeno1][jmeno2]=k
            tablenf[jmeno1][jmeno2]=len(np.isfinite(junk))
            g1=np.asarray(temp['g_x'])
            g2=np.asarray(temp['g_y'])
            g1red=g1*(1-redukce1*redukce[2]*(5-Q1))
            g2red=g2*(1-redukce1*redukce[2]*(5-Q2))
            mask = (g1 > mingroups) & (g2 > mingroups)
            junk=g1red[mask]/g2red[mask]
            k=np.mean(junk[np.isfinite(junk)])
            tablekg[jmeno1][jmeno2]=k
            tableng[jmeno1][jmeno2]=len(np.isfinite(junk))
            table_k=pd.concat([table_k, pd.DataFrame({'Datum':[startdat-W/2], 'Jmeno':[jmeno2], 'k_f':[kf], 'k_g':[k], 'Pruchod':[pruchod]})])
        # code.interact(local=locals())

        # posouvame se za hranici intervalu
        dat1=startdat-W-dW
        dat2=startdat-W
        # uriznout databazi jen na zvoleny casovy usek
        db2=db[(db['Datum'] >= dat1) & (db['Datum'] < dat2)]
        # najit pozorovatele, kteri se daji pouzit
        pozorovatele=db2['Pozorovatel']
        temp=list(pozorovatele.drop_duplicates())
        dostupni=copy.deepcopy(temp)
        dostupni=[jmeno for jmeno in dostupni if jmeno in referencni]
        # if (kralik in dostupni):
            # dostupni.remove(kralik)
        #if (kralik in dostupni) & (pruchod == 1):
        #    dostupni.remove(kralik)

        # code.interact(local=locals())
        for irow in range(0,len(db2)):
            date=db2.iloc[irow]['Datum']
            g=np.array(range(0,len(dostupni)))*np.nan
            f=np.array(range(0,len(dostupni)))*np.nan
            g_red=np.array(range(0,len(dostupni)))*np.nan
            f_red=np.array(range(0,len(dostupni)))*np.nan
            wg=np.array(range(0,len(dostupni)))*np.nan    
            wf=np.array(range(0,len(dostupni)))*np.nan 
            for iref in range(0,len(dostupni)):
                ref=dostupni[iref]
                if len(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['g']):
                    # pokud existuje porovnavaci udaj, tak nastav patricne hodnoty g, f a vah. Prepocitej g a f pres tabulku
                    g[iref]=np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['g'])*tablekg[kralik][ref]
                    f[iref]=np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['f'])*tablekf[kralik][ref]
                    g_red[iref]=g[iref]*(1-redukce2*redukce[2]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q'])))
                    f_red[iref]=f[iref]*(1-redukce2*redukce[1]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q'])))
                    # print((1-redukce[2]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q']))))
                    # print((1-redukce[1]*(5-np.asarray(db2[(db2['Pozorovatel']==ref) & (db2['Datum']==date)]['Q']))))
                    wg[iref]=tableng[kralik][ref]
                    wf[iref]=tablenf[kralik][ref]
            if pruchod > 1:
                gsave=(nansum(g_red*(wg/nansum(wg))))
                fsave=(nansum(f_red*(wf/nansum(wf))))
                sigma_g=nanstd(g_red)
                sigma_f=nanstd(f_red)
            else: 
                gsave=(nansum(g*(wg/nansum(wg))))
                fsave=(nansum(f*(wf/nansum(wf))))
                sigma_g=nanstd(g)
                sigma_f=nanstd(f)
            # gsave=(nansum(g*(wg/nansum(wg))))
            # fsave=(nansum(f*(wf/nansum(wf))))
            if np.isfinite(gsave):
                gsave=round(gsave)
            if np.isfinite(fsave):
                fsave=round(fsave)
            Rsave=10*gsave+fsave
            sigma_R=np.sqrt(10**2*sigma_g**2+sigma_f**2)
            if np.isfinite(Rsave):
                predikce_zpet=pd.concat([predikce_zpet, pd.DataFrame({'Datum':date, 'g':[gsave], 'f':[fsave], 'R':[Rsave], 'Q':[5]})])
                nejistota_zpet=pd.concat([nejistota_zpet, pd.DataFrame({'Datum':date, 'g':[sigma_g], 'f':[sigma_f], 'R':[sigma_R]})])

        # code.interact(local=locals())
        startdat=startdat-dW
        
    # code.interact(local=locals())
    # temp = pd.merge(predikce, predikce_zpet, how='inner', on=['Datum'])
    # junk = np.asarray(temp['R_y']-temp['R_x'])

    predikce=copy.deepcopy(predikce_zpet)
    nejistota=copy.deepcopy(nejistota_zpet)
    startdat=min(predikce['Datum'])

predikce.to_hdf('X_predikce.hdf','predikce', mode='w')
nejistota.to_hdf('X_nejistota.hdf','nejistota', mode='w')
table_k.to_hdf('X_predikce_tabulky_k.hdf','table_k', mode='w')

###########################################################
# porovnat se SIDC
sunspot_number = pd.read_hdf('SN.hdf','sunspot_number')
  
final_time=pd.date_range(start=min(predikce_zpet['Datum']), end=max(predikce_zpet['Datum']), freq='1D')
final_R=np.empty(len(final_time))*np.nan
final_sigmaR=np.empty(len(final_time))*np.nan
initial_R=np.empty(len(final_time))*np.nan
reference_R=np.empty(len(final_time))*np.nan
prumerne_kf=np.empty(len(final_time))*np.nan
prumerne_kg=np.empty(len(final_time))*np.nan
rozptyl_kf=np.empty(len(final_time))
rozptyl_kg=np.empty(len(final_time))

Whalf=dt.timedelta(days=13)

#import code
#code.interact(local=locals())
for iday in range(0,len(final_time)):
    datum=final_time[iday]
    temp=predikce_zpet[(predikce_zpet['Datum'] >= (datum-Whalf)) & (predikce_zpet['Datum'] <= (datum+Whalf))]
    final_R[iday]=np.mean(temp['R'])
    temp=nejistota_zpet[(nejistota_zpet['Datum'] >= (datum-Whalf)) & (nejistota_zpet['Datum'] <= (datum+Whalf))]
    final_sigmaR[iday]=np.mean(temp['R'])
    temp=initial[(initial['Datum'] >= (datum-Whalf)) & (initial['Datum'] <= (datum+Whalf))]
    initial_R[iday]=np.mean(temp['R'])
    temp=sunspot_number[(sunspot_number['Datum'] >= (datum-Whalf)) & (sunspot_number['Datum'] <= (datum+Whalf))]
    reference_R[iday]=np.mean(temp['R_i'])
    #
    temp=table_k[(table_k['Datum']==final_time[iday]) & (table_k['Pruchod']==Npruchodu)]
    if len(temp)>0:
        kf=np.asarray(temp['k_f'])
        kg=np.asarray(temp['k_g'])
        kf=kf[np.isfinite(kf)]
        kg=kg[np.isfinite(kg)]
        prumerne_kf[iday]=np.nanmean(kf)
        prumerne_kg[iday]=np.nanmean(kg)    
        rozptyl_kf[iday]=np.std(kf)
        rozptyl_kg[iday]=np.std(kg)    

with PdfPages('X_predikce.pdf') as pdf:   # jedno PDF s vice stranami
    # code.interact(local=locals())
    plt.figure()
    plt.plot(final_time, reference_R, 'r')
    plt.plot(final_time, final_R)
    plt.xlabel('Datum')
    plt.ylabel('Relativni cislo')
    plt.legend(['SIDC', 'Franta'])
    # plt.show()
    pdf.savefig()
    plt.close()
    #
    plt.figure()
    plt.plot(final_time, initial_R, 'r')
    plt.plot(final_time, final_R)
    win=np.isfinite(final_R) & np.isfinite(initial_R)
    C=np.corrcoef(initial_R[win], final_R[win])
    plt.title('$\\rho$ = '+str(C[0,1]))
    plt.xlabel('Datum')
    plt.ylabel('Relativni cislo')
    plt.legend(['Initial', 'Final'])
    #plt.show()
    pdf.savefig()
    plt.close()
    plt.figure()
    win=np.isfinite(final_R) & np.isfinite(reference_R)
    C=np.corrcoef(reference_R[win], final_R[win])
    plt.plot(reference_R, final_R, '.')
    plt.xlabel('Referencni R_i')
    plt.ylabel('Frantovo R')
    plt.title('Korelacni koeficient '+str(C[0,1]))
    # plt.show()
    pdf.savefig()
    plt.close()
    #
    for jmeno in referencni:
        temp=table_k[(table_k['Jmeno']==jmeno) & (table_k['Pruchod']==Npruchodu)]
        # temp=table_k[(table_k['Jmeno']==jmeno)]
        plt.figure()
        plt.plot(temp['Datum'], temp['k_f'], '--bo')
        plt.plot(temp['Datum'], temp['k_g'], '--ro')
        plt.legend(['$k_f$', '$k_g$'])
        plt.xlabel('Datum')
        plt.ylabel('Prevodni koeficienty pozorovatele')
        plt.title(jmeno)
        pdf.savefig()
        plt.close()
    #
    plt.figure()
    plt.errorbar(final_time, prumerne_kg, yerr=rozptyl_kg)
    plt.xlabel('Datum')
    plt.ylabel('Rozptyl prevodnich koeficientu pro $g$')
    pdf.savefig()
    plt.close()
    #
    plt.figure()
    plt.errorbar(final_time, prumerne_kf, yerr=rozptyl_kf)
    plt.xlabel('Datum')
    plt.ylabel('Rozptyl prevodnich koeficientu pro $f$')
    pdf.savefig()
    plt.close()

    
# import code
# code.interact(local=locals())

  #
# prumerne_kf=np.empty(len(final_time))*np.nan
# prumerne_kg=np.empty(len(final_time))*np.nan
# rozptyl_kf=np.empty(len(final_time))
# rozptyl_kg=np.empty(len(final_time))
# for iday in range(0,len(final_time)):
#     temp=table_k[(table_k['Datum']==final_time[iday]) & (table_k['Pruchod']==Npruchodu)]
#     if len(temp)>0:
#         kf=np.asarray(temp['k_f'])
#         kg=np.asarray(temp['k_g'])
#         kf=kf[np.isfinite(kf)]
#         kg=kg[np.isfinite(kg)]
#         prumerne_kf[iday]=np.nanmean(kf)
#         prumerne_kg[iday]=np.nanmean(kg)
#         rozptyl_kf[iday]=np.std(kf)
#         rozptyl_kg[iday]=np.std(kg)
    
# plt.errorbar(final_time, prumerne_kg, yerr=rozptyl_kg)
# plt.errorbar(final_time, prumerne_kf, yerr=rozptyl_kf)
    
 # plt.plot(final_time, prumerne_kg, '.')    