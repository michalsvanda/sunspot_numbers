# a module with useful routines needed to determine the optimal sunspot number

# OBSAHUJE

## create_database
# a routine that reads-in CSV-exported files from the program Slunce and forms a database in the Pandas dataframe
# format which is needed to perform the main.py code.

## initial_reference
# the function returns an initial series, which is taken as a series of some privileged observer

## initial_mean
# the function returns an initial series, which is computed as a mean over all observations. The mean is smoothed over 
# 28 days to ensure a reasonable coverage in time.


## initial flat
# FOR TESTING ONLY: the function returns the initial series, which is flat, but the time coverage corresponds to the 
# privileged observer


'''
The database format:
It is the Pandas DataFrame with the following items:
	
databaze={'Pozorovatel':[], 'Pristroj':[], 'Datum':[], 'UT':[], 'Q':[], 'Obr.':[], 'g':[], 'f':[], 'R':[]}

'Pozorovatel': STRING: The name of the observer
'Pristroj': STRING: A unique identification of the observer's instrument, it may also identify the station
'Datum': DATETIME timestamp: The date (the day) of the observation
'UT': FLOAT: the time of observation in UT in hours
'Q': FLOAT: the quality of observations, following the empirical table of visibility of structures
'Obr.': FLOAT: "obraz" (image), the another index assessing the seeing
'g': FLOAT: the total number of groups
'f': FLOAT: the total number a sunspots
'R': FLOAT: the relative number as 10g+f
 
 The mandatory fields are 'Pozorovatel', 'Pristroj', 'Datum', 'Q', 'g', 'f', 'R', other are optional

'''
import pandas as pd 
import datetime as dt
import code
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

def initial_mean(filename):
    # db = pd.read_hdf('databaze.hdf','db')
    db = pd.read_hdf(filename,'db')
    startdat=min(db['Datum'])
    #
    W=dt.timedelta(days=28) 
    dW=dt.timedelta(days=1) 
    #
    initial_series=pd.DataFrame({'Datum':[], 'g':[], 'f':[], 'R':[], 'Q':[]})
    initial_series['Datum']=initial_series['Datum'].astype('datetime64[ns]')
    while (startdat+W+dW) < max(db['Datum']):
        dat1=startdat
        # print(dat1, max(db['Datum']))
        dat2=startdat+W
        # cat the database to the investigate time range only
        db1=db[(db['Datum'] >= dat1) & (db['Datum'] < dat2)]
        middat=dat2+(dat1-dat2)/2
        # import code
        # code.interact(local=locals())
        gsave=np.nanmean(db1['g'])
        fsave=np.nanmean(db1['f'])
        Rsave=10.0*gsave+fsave
        Qsave=np.nanmean(db1['Q'])
        initial_series=pd.concat([initial_series, pd.DataFrame({'Datum':[middat], 'g':[gsave], 'f':[fsave], 'R':[Rsave], 'Q':[Qsave]})])
        #
        startdat=startdat+dW
    #
    return initial_series

def initial_reference(filename, observer):
    db = pd.read_hdf(filename,'db')
    dbX=db[db['Pozorovatel']==observer]
    initial_series=dbX.loc[:,['Datum', 'g','f','R','Q']]
    initial_series=initial_series.sort_values(by=['Datum'])
    return initial_series
    
def initial_flat(filename, observer):
    db = pd.read_hdf(filename,'db')
    dbX=db[db['Pozorovatel']==observer]
    initial_series=dbX.loc[:,['Datum', 'g','f','R','Q']]
    initial_series=initial_series.sort_values(by=['Datum'])
    initial_series['R']=30
    initial_series['g']=2
    initial_series['f']=10
    return initial_series
    
def create_database(filename, root_directory):
    # create_database('databaze.hdf', "./")
    # dir_in="./0_DATA/"
    dir_in=root_directory
    dirs = [f for f in listdir(dir_in) if isdir(join(dir_in, f))]

    # declare the global database
    databaze={'Pozorovatel':[], 'Pristroj':[], 'Datum':[], 'UT':[], 'Q':[], 'Obr.':[], 'g':[], 'f':[], 'R':[]}
    db=pd.DataFrame(databaze)
    for idir in dirs:
        dir_base=join(dir_in,idir)
        data_roky=[f for f in listdir(dir_base) if isdir(join(dir_base, f))]
        for drok in data_roky:
            dir_files=join(dir_base, drok)
            listfiles=[f for f in listdir(dir_files) if isfile(join(dir_files, f))]
            for infile in listfiles:
                file_in=join(dir_files, infile)
                print(file_in)
                # read-in CSV
                data = pd.read_csv(file_in,encoding='windows-1250', delimiter=';')
                #
                # cut only the first part of the table, which is crucial
                data=data[1:32]
                #
                # first read observers involved in the months. Initialise the DataFrame first
                observers={'No':[0], 'Name':['Unknown']}
                observers_df=pd.DataFrame(observers)
                #
                # pick up the data from CSV
                pozorovatele=(data[data.columns[19]])[9:30]
                # loop over the list, add to the dataframe
                iobs=1
                for poz in pozorovatele:
                  #print(poz.upper())
                  #print(iobs)
                  pozU=poz
                  if type(poz) == str:
                      pozU=poz.upper()
                  observers_df=pd.concat([observers_df, pd.DataFrame({'No':[iobs],'Name':[pozU]})], ignore_index=True)
                  #observers_df=pd.concat([observers_df, pd.DataFrame({'Name':[poz]}, index=iobs)])
                  iobs=iobs+1
                #
                # if we need to get the name according to the number, it is done line this: 
                #(observers_df.loc[observers_df['No'] == 1])['Name'].tolist()
                #
                mesic=int((data[data.columns[19]])[3])
                rok=int((data[data.columns[19]])[4])
                pristroj=' '
                temp=(data[data.columns[19]])[4:9].tolist()
                pristroj=pristroj.join(temp)
                #
                #forget about days with no observation
                valid=data[pd.notna(data['U'])]
                #
                #now loop over days and fill in the database
                for iday in range(0,len(valid)):
                  den=int(((valid.iloc[[iday]])['Den'].tolist())[0])
                  hour=int(((valid.iloc[[iday]])['U'].tolist())[0])
                  minute=int(((valid.iloc[[iday]])['T'].tolist())[0])
                  c_poz=int(((valid.iloc[[iday]])['Č.Poz.'].tolist())[0])
                  Q=int(((valid.iloc[[iday]])['Q'].tolist())[0])
                  g=int(((valid.iloc[[iday]])['g'].tolist())[0])
                  f=int(((valid.iloc[[iday]])['f'].tolist())[0])
                  R=int(((valid.iloc[[iday]])['r'].tolist())[0])
                  Obr=int(((valid.iloc[[iday]])['Obr.'].tolist())[0])
                  datum=dt.datetime(year=rok, month=mesic, day=den)
                  db=pd.concat([db, pd.DataFrame({'Pozorovatel':(observers_df.loc[observers_df['No'] == c_poz])['Name'].tolist(), 'Pristroj':pristroj, 'Datum':[datum], 'UT':[hour+minute/60], 'Q':[Q], 'Obr.':[Obr], 'g':[g], 'f':[f], 'R':[R]})], ignore_index=True)
                  #
                  #
    for jmeno in db['Pozorovatel'].unique().tolist():
        print(jmeno, len(db[db['Pozorovatel']==jmeno]))
        #
    db.to_hdf(filename,'db', mode='w')


