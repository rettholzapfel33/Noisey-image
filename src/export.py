import os
from os import access, R_OK
from os.path import isfile
import pandas as pd

# Column titles for the dataframes
indexes = ['Class', 'Confidence', 'Top X', 'Top Y', 'Bottom X', 'Bottom Y']

#df = pd.read_csv('exp_4/WebP_Compression_0/original.txt', sep=" ", engine='python')

#df.columns = indexes

directory = 'data/tmp/runs/'

# Loop through the runs directories
for subdir, dirs, files in os.walk(directory):

    for filename in files:

        # Get the file path
        filepath = subdir + os.sep + filename

        if filepath.endswith(".txt"):

            # Check if the file has content
            if os.stat(filepath).st_size != 0:

                # Convert text file into dataframe
                df = pd.read_csv(filepath, sep=" ", engine='python')

                df.columns = indexes

                toCSV = subdir + os.sep + 'original.csv'

                # Export to csv
                df.to_csv(toCSV, index=False)
