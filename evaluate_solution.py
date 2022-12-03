#!/usr/bin/python

import sys, getopt
from PlateReader import PlateReader
import cv2
import os
import pandas as pd
import datetime
import tqdm
from tqdm.contrib import tenumerate

def main(argv):
    inputdir = ''
    outputfile = ''

    user_manual = 'evaluate_solution.py -i <inputdir> (optional) -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["idir=","ofile="])
    except getopt.GetoptError:
        print(user_manual)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(user_manual)
            sys.exit()
        elif opt in ("-i", "--idir"):            
            inputdir = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    
    if inputdir == '':
        print(user_manual)
        sys.exit(2)

    if not os.path.isdir(inputdir):
        print('Input directory does not exist.')
        sys.exit(2)

    print('Input directory is ', inputdir)
    
    print('Initializing PlateReader...')
    plate_reader = PlateReader()

    # create dataframe to store results with filename, plates
    df = pd.DataFrame(columns=['filename', 'plates'])

    print('Starting to read images...')
    # iterate through all the images in the input directory
    for index, filename in tenumerate(os.listdir(inputdir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # read the image
            img = cv2.imread(os.path.join(inputdir, filename))
            # read the plate
            plates = plate_reader.read(img)['plates']
            # add the results to the dataframe
            df = df.append({'filename': filename, 'plates': ';'.join(plates)}, ignore_index=True)
        

    print('Saving results to file...')
    if outputfile == '':
        # set the output file name to results + timestamp
        outputfile = 'results_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv'

    # save the results to a csv file
    df.to_csv(outputfile, index=False)
    print('Output file is ', outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])

