# sunspot_numbers

Code to construct sunspot number series from the series recorded by many observers with their different personal/instrumental biases

## FILES IN REPOSITORY

### main.py
The main module.

### sunspots.py
A module with useful routines needed to determine the optimal sunspot number, including the reading routine for the CSV export from Slunce software for the CESLOPOL observers.

### conf.py
Contains the configuration directives influencing the performance of the code

### plot.py
A pseudocode showing how to do some trivial plots and comparisons of the results

### databaze.hdf
A compiled database used in the study submitted to Solar Physics journal, it contains example observations. 

### SN.hdf
WDC-SILSO 2.0 sunspot number

### GN.hdf
Revised group number

### observations_to_database.py
A mock-up script (pseudo-code) to demonstrate how to fill in the database Pandas DataFrame compatible with the main code. 

## DATA FORMATS
### observations
```
databaze={'Pozorovatel':[], 'Pristroj':[], 'Datum':[], 'UT':[], 'Q':[], 'Obr.':[], 'g':[], 'f':[], 'R':[]}
```

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

### target series
```
target={'Datum':[], 'g':[], 'f':[], 'R':[], 'Q':[5]}
```

'Datum': DATETIME timestamp: The date (the day) of the observation
'g': FLOAT: the total number of groups
'f': FLOAT: the total number a sunspots
'R': FLOAT: the relative number as 10g+f
'Q': FLOAT: the quality of observations, default is 5 in the case of reduction to optimal conditions

### uncertainties
```
uncertainty={'Datum':[], 'g':[], 'f':[], 'R':[]}
```

'Datum': DATETIME timestamp: The date (the day) of the observation
'g': FLOAT: uncertainty of the total number of groups
'f': FLOAT: uncertainty of the total number a sunspots
'R': FLOAT: uncertainty the relative number as 10g+f

### conversion coefficient tables
```
k_table={'Datum':[], 'Jmeno':[], 'k_f':[], 'k_g':[], 'Pruchod':[]}
```
'Datum': DATETIME timestamp: The date (the day) of the observation
'Jmeno': STRING: the name of the observer
'k_f': FLOAT: observer's conversion coefficient to targer series for f (total sunspots number)
'k_g': FLOAT: observer's conversion coefficient to targer series for g (number of sunspot groups)
'Pruchod': FLOAT: identification of the iteration for which these coefficients are valid. Integer
                  values indicate the full-interation (forth and back), whereas the values ending with 
				  0.5 indicate the half-iteration (forth only)
				  