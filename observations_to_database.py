# this is a mock-up script to demonstrate how to create a valid form of the database accepted by the main code

import pandas as pd 
import datetime as dt

# declare the DataFrame for the database
db={'Pozorovatel':[], 'Pristroj':[], 'Datum':[], 'UT':[], 'Q':[], 'Obr.':[], 'g':[], 'f':[], 'R':[]}

# read-in the observations, loop over all files/records
for file in list_of_files_with_observations:
    observation=parse_file(file)  # read the content of the file. Here we expect that this will be a structured record

    year=observation.year  # year (integer)
    month=observation.month  # month (integer)
    day=observation.day  # day (integer)
    time=observation.time_UT  # UT hours (float)
    observer=observation.observer  # observer name/unique identifier (string)
    instr=observation.intr # identification of the instrument/telescope (string)
    quality=observation.quality # subjective quality assessment of the observation (integer)
    image_quality=observation.image_quality # alternative quality assessment, this one is not used in the code
    g=observation.g # number of sunspot groups (integer)
    f=observation.f # number of sunspots (integer)
    R=observation.R # relative number (integer)

    date=dt.datetime(year=year, month=month, day=day)

    db=pd.concat([db, pd.DataFrame({'Pozorovatel':observer, 'Pristroj':instr, 'Datum':[date], 'UT':[time], 
                                    'Q':[quality], 'Obr.':[image_quality], 'g':[g], 'f':[f], 'R':[R]})], ignore_index=True)

# store the database to HDF file                                    
db.to_hdf(filename,'db', mode='w')