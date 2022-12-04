import pandas as pd
from urllib import request, error
import urllib.request
import os
from tqdm import tqdm
# read csv file
df = pd.read_csv('KF_HF_database.csv', sep=';', header=None)

# read user input
user_input = input('Enter the number of the dataset you want to download \n License plates: 1 \n High res pictures: 2 \n Low res pictures: 3 \n ')

if (int(user_input) == 1):
    read_column = 2
    directory_name = 'license_plate_images'
elif (int(user_input) == 2):
    read_column = 3
    directory_name = 'high_res_images'
elif (int(user_input) == 3):
    read_column = 4
    directory_name = 'low_res_images'
else:
    print('Invalid input')
    exit()

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0'), ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8')]
urllib.request.install_opener(opener)

#iterate over rows with iterrows()
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # download the file from the url 4. column
    url = row[read_column]
    # save the file under the name of the 1. column
    filename = row[0]
    # create directory if not exists
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, directory_name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
       
    try:
        urllib.request.urlretrieve(url, filename=directory_name  + "/" + filename + ".jpg")
    except error.HTTPError as e:
        print(e.fp.read())
        print("error: " + filename + " " + str(index) + " " + url)

    


