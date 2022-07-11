#
# PLOTTING, COMPARSION etc 
#
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import datetime as dt
import numpy as np

from conf import config

###########################################################
def do_plots():
    '''
    Do some trivial plots of the results    
    '''
    # read in the files first
    target = pd.read_hdf('target.hdf','target')
    initial = pd.read_hdf('initial.hdf','initial')
    uncertainty = pd.read_hdf('uncertainty.hdf','uncertainty')
    table_k = pd.read_hdf('k_tables.hdf','table_k')
    observers_considered=table_k['Jmeno'].drop_duplicates()

    # import code
    # code.interact(local=locals())
    # SILSO v2
    sunspot_number = pd.read_hdf('SN.hdf','sunspot_number')

    # now interpolate all series to the same (regular) grid in time
    final_time=pd.date_range(start=min(target['Datum']), end=max(target['Datum']), freq='1D')
    final_R=np.empty(len(final_time))*np.nan
    final_sigmaR=np.empty(len(final_time))*np.nan
    initial_R=np.empty(len(final_time))*np.nan
    reference_R=np.empty(len(final_time))*np.nan
    prumerne_kf=np.empty(len(final_time))*np.nan
    prumerne_kg=np.empty(len(final_time))*np.nan
    rozptyl_kf=np.empty(len(final_time))
    rozptyl_kg=np.empty(len(final_time))

    # use the average over 13 days
    Whalf=dt.timedelta(days=13)

    #loop to fix the time axis, average over 13 days both
    for iday in range(0,len(final_time)):
        datum=final_time[iday]
        temp=target[(target['Datum'] >= (datum-Whalf)) & (target['Datum'] <= (datum+Whalf))]
        final_R[iday]=np.mean(temp['R'])
        temp=uncertainty[(uncertainty['Datum'] >= (datum-Whalf)) & (uncertainty['Datum'] <= (datum+Whalf))]
        final_sigmaR[iday]=np.mean(temp['R'])
        temp=initial[(initial['Datum'] >= (datum-Whalf)) & (initial['Datum'] <= (datum+Whalf))]
        initial_R[iday]=np.mean(temp['R'])
        temp=sunspot_number[(sunspot_number['Datum'] >= (datum-Whalf)) & (sunspot_number['Datum'] <= (datum+Whalf))]
        reference_R[iday]=np.mean(temp['R_i'])
        #
        temp=table_k[(table_k['Datum']==final_time[iday]) & (table_k['Pruchod']==config["Niterations"])]
        if len(temp)>0:
            kf=np.asarray(temp['k_f'])
            kg=np.asarray(temp['k_g'])
            kf=kf[np.isfinite(kf)]
            kg=kg[np.isfinite(kg)]
            prumerne_kf[iday]=np.nanmean(kf)
            prumerne_kg[iday]=np.nanmean(kg)    
            rozptyl_kf[iday]=np.std(kf)
            rozptyl_kg[iday]=np.std(kg)    

    # plot an output file, PDF with pages
    with PdfPages('plots.pdf') as pdf:   # jedno PDF s vice stranami
        # code.interact(local=locals())
        plt.figure()
        plt.plot(final_time, reference_R, 'r')
        plt.plot(final_time, final_R)
        plt.xlabel('Date')
        plt.ylabel('Relative number')
        plt.legend(['SIDC', 'our'])
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
        plt.xlabel('Date')
        plt.ylabel('Relative number')
        plt.legend(['Initial', 'Final'])
        #plt.show()
        pdf.savefig()
        plt.close()
        plt.figure()
        win=np.isfinite(final_R) & np.isfinite(reference_R)
        C=np.corrcoef(reference_R[win], final_R[win])
        plt.plot(reference_R, final_R, '.')
        plt.xlabel('SILSO')
        plt.ylabel('Our')
        plt.title('Correlation coefficient '+str(C[0,1]))
        # plt.show()
        pdf.savefig()
        plt.close()
        #
        for jmeno in observers_considered:
            temp=table_k[(table_k['Jmeno']==jmeno) & (table_k['Pruchod']==config["Niterations"])]
            # temp=table_k[(table_k['Jmeno']==jmeno)]
            plt.figure()
            plt.plot(temp['Datum'], temp['k_f'], '--bo')
            plt.plot(temp['Datum'], temp['k_g'], '--ro')
            plt.legend(['$k_f$', '$k_g$'])
            plt.xlabel('Date')
            plt.ylabel('Conversion coefficients of the observer')
            plt.title(jmeno)
            pdf.savefig()
            plt.close()
        #
        plt.figure()
        plt.errorbar(final_time, prumerne_kg, yerr=rozptyl_kg)
        plt.xlabel('Datum')
        plt.ylabel('Spread of the conversion coefficients for $g$')
        pdf.savefig()
        plt.close()
        #
        plt.figure()
        plt.errorbar(final_time, prumerne_kf, yerr=rozptyl_kf)
        plt.xlabel('Datum')
        plt.ylabel('Spread of the conversion coefficients for  $f$')
        pdf.savefig()
        plt.close()
