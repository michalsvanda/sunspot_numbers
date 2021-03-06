'''
The suggested algorithm written in Czech follows. 

for English refer to the paper

Śvanda, Pavelková, Dvořák, Solarová: Iterative construction of the optimal sunspot number series, submitted to Solar Physics

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

####################################
# IMPORTS 
#
import pandas as pd 
import datetime as dt
from os import listdir
from os.path import isfile, join, isdir
import code
import numpy as np

import copy

####################################
# PARAMETERS
#

from conf import config

# read in the database file

db = pd.read_hdf(config["database_file"],'db')

####################################
# DECLARATIONS 
#


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

####################################
# FUNCTIONAL CODE

####################################

def evaluate_coefficients(database, dat1, dat2, target, config, initial_observer, observers_considered, table_k, direction):
    '''
    Fuction that evaluates the personal scaling coefficients of the observers with respect to the target series in an evaluation
    windows (which has a width W). 
    
    Inputs:
    database: pandas dataframe with the database of the observations in the required format
    dat1: starting date of the evaluation interval, in datetime format
    dat2: final date of the evaluation interval, in datetime format
    target: the target series, pandas dataframe in the fixed format
    config: global parameters, as dictionary
    initial_observer: the name of the initial observer as string. It only serves to evaluate the table columns
    observers_considered: considered observers for the weights, list of strings
    table_k: pandas dataframe with a fixed format for storing the coefficient values for the tuning/debugging/control purposes
    direction: the direction of the run. +1 for direction along the time arrow, -1 for direction against the arrow of time 
    
    Outputs:
    tablekg: pandas dataframe with personal coefficients for g
    tablekf: pandas dataframe with personal coefficients for f
    tableng: pandas dataframe with number of pairs to evaluate the coefficient for g
    tablenf: pandas dataframe with number of pairs to evaluate the coefficient for f
    '''
    # check some trivial validity of the input parameters
    valid_direction = {1, -1}
    if direction not in valid_direction:
        raise ValueError("Error: direction must be one of %r." % valid_direction)
    if dat1 >= dat2:
        raise ValueError("Error: dat2 must be larger than dat1")
    # cut the database only to W-long window, respect the open or closed boudaries according to the direction
    if direction == 1:
        db_cut=database[(database['Datum'] >= dat1) & (database['Datum'] < dat2)]
    else:
        db_cut=database[(database['Datum'] > dat1) & (database['Datum'] <= dat2)]

    # compute coefficients for the prediction
    tablekf=pd.DataFrame(columns=[initial_observer], index=observers_considered)
    tablekg=pd.DataFrame(columns=[initial_observer], index=observers_considered)
    tablenf=pd.DataFrame(columns=[initial_observer], index=observers_considered)
    tableng=pd.DataFrame(columns=[initial_observer], index=observers_considered)
    name1=initial_observer
    # subset of the initial series
    set1=target[(target['Datum'] >= dat1) & (target['Datum'] < dat2)]
    for name2 in observers_considered:
        # subset of the other used observer
        set2=db_cut[(db_cut['Pozorovatel'] == name2)]   
        # intersection of both on date  
        sets_intersection = pd.merge(set1, set2, how='inner', on=['Datum'])  # slucujeme po poli datum
        Q1=np.asarray(sets_intersection['Q_x'])
        Q2=np.asarray(sets_intersection['Q_y'])
        f1=np.asarray(sets_intersection['f_x'])
        f2=np.asarray(sets_intersection['f_y'])
        # reduction to optimal observing conditions
        f1red=f1*(1-config["Qreduction_eval_switch"]*config["Qreduction_slopes"][1]*(5-Q1))
        f2red=f2*(1-config["Qreduction_eval_switch"]*config["Qreduction_slopes"][1]*(5-Q2))
        # limiting to only observations above the thresholds
        mask = (f1 > config["minspots"]) & (f2 > config["minspots"])
        # conversion coefficient for f (count of sunspots)
        ratio_=f1red[mask]/f2red[mask]    
        tablekf[name1][name2]=np.mean(ratio_[np.isfinite(ratio_)])  # coefficient
        tablenf[name1][name2]=len(np.isfinite(ratio_))   # number of applicable pairs (to be used in weighting)
        g1=np.asarray(sets_intersection['g_x'])
        g2=np.asarray(sets_intersection['g_y'])
        g1red=g1*(1-config["Qreduction_eval_switch"]*config["Qreduction_slopes"][2]*(5-Q1))
        g2red=g2*(1-config["Qreduction_eval_switch"]*config["Qreduction_slopes"][2]*(5-Q2))
        mask = (g1 > config["mingroups"]) & (g2 > config["mingroups"])
        # conversion coefficient for g (count of groups)
        ratio_=g1red[mask]/g2red[mask]
        tablekg[name1][name2]=np.mean(ratio_[np.isfinite(ratio_)])  # coefficient
        tableng[name1][name2]=len(np.isfinite(ratio_))  # number of applicable pairs (to be used in weighting)
        if direction == 1:
            # moving along the arrow of time, this is a half-iteration
            iteration_shift=-0.5
        else:
            # moving against the arrow of time, this is a full-iteration
            iteration_shift=0
        table_k=pd.concat([table_k, pd.DataFrame({'Datum':[startdat+direction*config["W"]/2], 'Jmeno':[name2], 'k_f':tablekf[name1][name2], 'k_g':tablekg[name1][name2], 'Pruchod':[iteration+iteration_shift]})])
    # code.interact(local=locals())
    return tablekg, tablekf, tableng, tablenf, table_k

####################################

####################################

def predict_target(database, dat1, dat2, target, uncertainty, config, initial_observer, observers_considered, tablekg, tablekf, tableng, tablenf, direction): 
    '''
    Function to predict the targer series from the series of other observers 
        
    Inputs:
    database: pandas dataframe with the database of the observations in the required format
    dat1: starting date of the evaluation interval, in datetime format
    dat2: final date of the evaluation interval, in datetime format
    target: the target series, pandas dataframe in the fixed format
    uncertainty: uncertainties of the target series, pandas dataframe in the fixed format
    config: global parameters, as dictionary
    initial_observer: the name of the initial observer as string. It only serves to evaluate the table columns
    observers_considered: considered observers for the weights, list of strings
    tablekg: pandas dataframe with personal coefficients for g
    tablekf: pandas dataframe with personal coefficients for f
    tableng: pandas dataframe with number of pairs to evaluate the coefficient for g
    tablenf: pandas dataframe with number of pairs to evaluate the coefficient for f
    direction: the direction of the run. +1 for direction along the time arrow, -1 for direction against the arrow of time 
    
    Outputs:
    target: the target series, pandas dataframe in the fixed format
    uncertainty: uncertainties of the target series, pandas dataframe in the fixed format    
    '''
    # check some trivial validity of the input parameters
    valid_direction = {1, -1}
    if direction not in valid_direction:
        raise ValueError("Error: direction must be one of %r." % valid_direction)
    if dat1 >= dat2:
        raise ValueError("Error: dat2 must be larger than dat1")
    # cut the database only to W-long window, respect the open or closed boudaries according to the direction
    if direction == 1:
        db_cut=database[(database['Datum'] >= dat1) & (database['Datum'] < dat2)]
    else:
        db_cut=database[(database['Datum'] > dat1) & (database['Datum'] <= dat2)]
    # find usable observers in the prediction window
    observers_with_duplicates=db_cut['Pozorovatel']
    observers_without_duplicates=list(observers_with_duplicates.drop_duplicates())
    available_observers_for_prediction=copy.deepcopy(observers_without_duplicates)
    available_observers_for_prediction=[name for name in available_observers_for_prediction if name in observers_considered] # check if their observations are available
    #import code
    #code.interact(local=locals())
    print(available_observers_for_prediction)    
    if len(available_observers_for_prediction) != 0:
        # there are some available observers, continue
        # code.interact(local=locals())
        for irow in range(0,len(db_cut)):  # go day by day
            date=db_cut.iloc[irow]['Datum']
            # first fill the expected arrays with NaNs
            g=np.array(range(0,len(available_observers_for_prediction)))*np.nan
            f=np.array(range(0,len(available_observers_for_prediction)))*np.nan
            g_red=np.array(range(0,len(available_observers_for_prediction)))*np.nan
            f_red=np.array(range(0,len(available_observers_for_prediction)))*np.nan
            # weights
            wg=np.array(range(0,len(available_observers_for_prediction)))*np.nan    
            wf=np.array(range(0,len(available_observers_for_prediction)))*np.nan 
            for iref in range(0,len(available_observers_for_prediction)):  # go observer by observer and compute individual predictions
                reference_observer=available_observers_for_prediction[iref]
                if len(db_cut[(db_cut['Pozorovatel']==reference_observer) & (db_cut['Datum']==date)]['g']):
                    # in the case there exists an observation, get g and f values and recompute them to the target using the g and f table
                    g[iref]=np.asarray(db_cut[(db_cut['Pozorovatel']==reference_observer) & (db_cut['Datum']==date)]['g'])*tablekg[initial_observer][reference_observer]
                    f[iref]=np.asarray(db_cut[(db_cut['Pozorovatel']==reference_observer) & (db_cut['Datum']==date)]['f'])*tablekf[initial_observer][reference_observer]
                    g_red[iref]=g[iref]*(1-config["Qreduction_predict_switch"]*config["Qreduction_slopes"][2]*
                                (5-np.asarray(db_cut[(db_cut['Pozorovatel']==reference_observer) & (db_cut['Datum']==date)]['Q'])))
                    f_red[iref]=f[iref]*(1-config["Qreduction_predict_switch"]*config["Qreduction_slopes"][1]*
                                (5-np.asarray(db_cut[(db_cut['Pozorovatel']==reference_observer) & (db_cut['Datum']==date)]['Q'])))
                    # compute the weights (number of pairs used for evaluation of the conversion coefficients)
                    wg[iref]=tableng[initial_observer][reference_observer]
                    wf[iref]=tablenf[initial_observer][reference_observer]
            if iteration > 1:
                gsave=(nansum(g_red*(wg/nansum(wg))))
                fsave=(nansum(f_red*(wf/nansum(wf))))
                sigma_g=nanstd(g_red)
                sigma_f=nanstd(f_red)
            else: 
                gsave=(nansum(g*(wg/nansum(wg))))
                fsave=(nansum(f*(wf/nansum(wf))))
                sigma_g=nanstd(g)
                sigma_f=nanstd(f)
            # round-off the values    
            if np.isfinite(gsave):
                gsave=round(gsave)
            if np.isfinite(fsave):
                fsave=round(fsave)
            # compute R (either finite or NaN)
            Rsave=10*gsave+fsave
            # computer uncertainty of R (either finite or NaN)
            sigma_R=np.sqrt(10**2*sigma_g**2+sigma_f**2)
            # store those values, where R is finite
            if np.isfinite(Rsave):
                target=pd.concat([target, pd.DataFrame({'Datum':date, 'g':[gsave], 'f':[fsave], 'R':[Rsave], 'Q':[5]})])
                uncertainty=pd.concat([uncertainty, pd.DataFrame({'Datum':date, 'g':[sigma_g], 'f':[sigma_f], 'R':[sigma_R]})])
        #
    return target, uncertainty
####################################

#
# PREPARATIONS

# prepare the list of available observers:
# pre-sort observers and chose only their unique names. Those are stored in the list "observers"
observers_with_duplicates=db['Pozorovatel']
observers=list(observers_with_duplicates.drop_duplicates())
print(len(observers))

# preselection of the observers that have at least 100 observations each. Those are in the list "observers_long_term"
# these are not used further by default, however they might be useful 

observers_long_term=copy.deepcopy(observers)
for pozorovatel in observers:
    print(pozorovatel, ' ', len(db[db['Pozorovatel']==pozorovatel]))
    if len(db[db['Pozorovatel']==pozorovatel]) < 100:
        observers_long_term.remove(pozorovatel)
        
# cut all observers except for those long-term with more than 100 records
db=db[db['Pozorovatel'].apply((lambda x: x in observers_long_term) )]


# store the available observers (unique names) into yet another list

observers_considered=copy.deepcopy(observers)
#observers_considered=copy.deepcopy(observers_long_term)

# initiate the table of personal coefficients

table_k=pd.DataFrame({'Datum':[], 'Jmeno':[], 'k_f':[], 'k_g':[], 'Pruchod':[]})


##############################
# INITIAL MODEL

###
# 1. a reference observer

# start from a named reference observer. Here, e.g., 'ZLOCH FRANTIŠEK'
reference_observer='ZLOCH FRANTIŠEK'
from sunspots import initial_reference
target=initial_reference(config["database_file"], reference_observer)

###
# 2. a flat initial series

# from sunspots import initial_flat
# target=initial_flat(config["database_file"], reference_observer)

# variable "initial_observer" contains the name of the initial observer. 
initial_observer=reference_observer

###
# 3. a plain average (arithmentic mean across the observers)
# from sunspots import initial_mean
# target=initial_mean(config["database_file"])
# initial_observer='AVERAGE'


# save the initial series, this is useful for the comparisons or debugging for later
initial=copy.deepcopy(target)  # initial series is only stored to file at the end

# starting date 
startdat=min(target['Datum'])

# Cut the initial series only to the first W-long interval
dat1=startdat
dat2=startdat+config["W"]
target=target[(target['Datum'] >= dat1) & (target['Datum'] < dat2)]

#######################################

# MAIN CYCLE 

#######################################


##########
# Uncertainties 
# Uncertainty of function f(x_1, x_2, ...) with independent variables x_1, ... and their uncertainties sigma_x1, ... is given as
# sigma_f = sqrt{ suma_i [ (\partial f / \partial x_i)^2 sigma_xi^2 ] }
# hence for R = 10*g + f
# sigma_R = sqrt{ 10^2*sigma_g^2 + sigma_f^2 }
#
# first copy the existing structure for consistency
uncertainty=copy.deepcopy(target)
# then zero it
uncertainty['g']=0
uncertainty['f']=0
uncertainty['R']=0
# remove Q column, which is useless
uncertainty=uncertainty.drop(columns=['Q'])

# import code
# code.interact(local=locals())

# cycle variable "iteration" contains the number of full iteration passed
for iteration in range(1,config["Niterations"]+1):
    #observers_considered=copy.deepcopy(observers)
    # if iteration == 1:
        # observers_considered.remove(initial_observer)
    # observers_considered.remove(initial_observer)
    print('*********************')
    print('Pass No. '+str(iteration))
    print('Following the arrow of time...')
    while (startdat+config["W"]+config["dW"]) < max(db['Datum']):
        dat1=startdat
        print(dat1, max(db['Datum']))
        dat2=startdat+config["W"]
        tablekg, tablekf, tableng, tablenf, table_k = evaluate_coefficients(database=db, dat1=dat1, dat2=dat2, target=target, 
                                                                            config=config, initial_observer=initial_observer, 
                                                                            observers_considered=observers_considered, table_k=table_k,
                                                                            direction=1)
        # shift beyond the evaluation interval
        dat1=startdat+config["W"]
        dat2=startdat+config["W"]+config["dW"]
        target, uncertainty = predict_target(database=db, dat1=dat1, dat2=dat2, target=target, uncertainty=uncertainty, config=config,
                                             initial_observer=initial_observer, observers_considered=observers_considered, tablekg=tablekg,
                                             tablekf=tablekf, tableng=tableng, tablenf=tablenf, direction=1)
        # move on to the next evaluation
        startdat=startdat+config["dW"]

    ###############################################################################

    # code.interact(local=locals())

    #######################################

    # Main cycle, going backwards

    #######################################
    print('Against the arrow of time...')
    startdat=max(target['Datum'])
    # copy the structures first 
    target_backwards=target[(target['Datum'] <= startdat) & (target['Datum'] > (startdat-config["W"]))]
    uncertainty_backwards=uncertainty[(uncertainty['Datum'] <= startdat) & (uncertainty['Datum'] > (startdat-config["W"]))]
    while (startdat-config["W"]-config["dW"]) > min(db['Datum']):
        dat1=startdat-config["W"]
        print(dat1, min(db['Datum']))
        dat2=startdat
        tablekg, tablekf, tableng, tablenf, table_k = evaluate_coefficients(database=db, dat1=dat1, dat2=dat2, target=target_backwards,
                                                                            config=config, initial_observer=initial_observer,
                                                                            observers_considered=observers_considered, table_k=table_k,
                                                                            direction=-1)
        # shift (backwards) beyond the evaluation interval
        dat1=startdat-config["W"]-config["dW"]
        dat2=startdat-config["W"]
        target_backwards, uncertainty_backwards = predict_target(database=db, dat1=dat1, dat2=dat2, target=target_backwards, uncertainty=uncertainty_backwards,
                                                                 config=config, initial_observer=initial_observer, observers_considered=observers_considered,
                                                                 tablekg=tablekg, tablekf=tablekf, tableng=tableng, tablenf=tablenf, direction=1)
        # move on (backwards) to the next evaluation
        startdat=startdat-config["dW"]
        
    # the cycle is finished. Copy structures to go for the next cycle or for plotting/saving/etc.
    target=copy.deepcopy(target_backwards)
    uncertainty=copy.deepcopy(uncertainty_backwards)
    startdat=min(target['Datum'])

# store to the files
# the series
target.to_hdf('target.hdf','target', mode='w')
# its uncertainty
uncertainty.to_hdf('uncertainty.hdf','uncertainty', mode='w')
# conversion coefficients
table_k.to_hdf('k_tables.hdf','table_k', mode='w')
# initial series
initial.to_hdf('initial.hdf','initial', mode='w')

####
# 
# THE CODE IS DONE HERE. WHAT FOLLOWS IS PURELY OPTIONAL
#
####
#
# PLOTTING, COMPARSION etc 
if config["do_plot_results"]:
    from plot import do_plots
    do_plots()
    
