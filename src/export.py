import os
from os import access, R_OK
from os.path import isfile
import pandas as pd

def main():

    current = os.getcwd()

    # Column titles for the dataframes
    indexes = ['Class', 'Confidence', 'Top X', 'Top Y', 'Bottom X', 'Bottom Y']

    #df = pd.read_csv('data/tmp/runs/exp_2/Gaussian_Noise_JPEG_Compression_Salt_and_Pepper_2/original.txt', sep=" ", engine='python')

    #df.columns = indexes

    # Loop through the runs directories
    for subdir, dirs, files in os.walk(current):

        for directory in dirs:

            # If in /runs directory
            if directory == 'runs':

                # Loop through this directory until text file is found
                for subdir2, dirs2, files2 in os.walk(subdir):

                    for filename in files2:

                        # Get the file path
                        filepath = subdir2 + os.sep + filename

                        if filepath.endswith(".txt"):

                            # Check if the file has content
                            if os.stat(filepath).st_size != 0:

                                # Convert text file into dataframe
                                try:
                                    df = pd.read_csv(filepath, sep=" ", engine='python')

                                except pd.errors.ParserError:

                                    continue

                                df.columns = indexes

                                toCSV = subdir2 + os.sep + 'original.csv'

                                # Export to csv
                                df.to_csv(toCSV, index=False)

main()