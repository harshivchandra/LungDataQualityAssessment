##import os
##
##print(os.listdir("NIHChestXray"))
##path ="NIHChestXray/"
##x = ['images_001', 'images_002', 'images_003', 'images_004', 'images_005', 'images_006', 'images_007', 'images_008', 'images_009', 'images_010', 'images_011', 'images_012']
##toappend = "/images/"
##
##x = [str(path+i+toappend)for i in x]
##
##final_path = path+"images/"
##
##
##for i in x:
##    for j in os.listdir(i):
##        os.rename(i+"/"+j,final_path+j)
##        


import pydicom
from PIL import Image
import numpy as np

def dicom_to_png(dicom_file_path, png_file_path):
    # Load the DICOM image
    ds = pydicom.dcmread(dicom_file_path)

    # Get the pixel array from the DICOM dataset
    pixel_array = ds.pixel_array

    # Convert to a format that can be saved by PIL
    # Normalize the pixel values to be between 0 and 255
    pixel_array = pixel_array - np.min(pixel_array)
    if np.max(pixel_array) != 0:
        pixel_array = pixel_array / np.max(pixel_array)
    pixel_array = (pixel_array * 255).astype(np.uint8)

    # Create an Image object from the array
    image = Image.fromarray(pixel_array)

    # Save the image as PNG
    image.save(png_file_path)
####
##### Example usage
##dicom_file_path = "D:/Projects/Code/Lung Disease Prediction/input/CheXpert-v1.0-small/Lung-PET-CT-Dx/Input_data/manifest-1608669183333/Lung-PET-CT-Dx/Lung_Dx-A0002/04-25-2007-NA-ThoraxAThoraxRoutine Adult-34834/3.000000-ThoraxRoutine  8.0.0  B40f-10983/1-01.dcm"
##png_file_path = "file.png"
##dicom_to_png(dicom_file_path, png_file_path)
import pandas as pd
import os

##def generate_one_hot_encoded_labels(metadata_df):
##    # Mapping of letters to cancer types
##    cancer_mapping = {
##        'A': 'Adenocarcinoma',
##        'B': 'Small Cell Carcinoma',
##        'E': 'Large Cell Carcinoma',
##        'G': 'Squamous Cell Carcinoma'
##    }
##
##    # Initialize columns for each cancer type
##    for cancer_type in cancer_mapping.values():
##        metadata_df[cancer_type] = 0
##
##    # Iterate over rows and set the appropriate column to 1 based on the Subject ID
##    for index, row in metadata_df.iterrows():
##        subject_id = row['Subject ID']
##        for letter, cancer_type in cancer_mapping.items():
##            if letter in subject_id:
##                metadata_df.at[index, cancer_type] = 1
##
##    return metadata_df
##
##metadata_df = pd.read_csv("D:\Projects\Code\Lung Disease Prediction\input\CheXpert-v1.0-small\Lung-PET-CT-Dx\Input_data\manifest-1608669183333\metadata.csv")
##
### Apply the function to the metadata dataframe
##metadata_with_labels = generate_one_hot_encoded_labels(metadata_df)
##
### Display the first few rows with the new one-hot encoded labels
####metadata_with_labels.head()
##import numpy as np
##path = "Lung-PET-CT-Dx/Input_data/manifest-1608669183333/"
##df = pd.read_csv("metadata_dicom.csv")
##m = df[df.columns[0]].values
##ss = []
##for i in m:
##    ss.append(path+i.strip(".\\").replace("\\","/"))
##ss = np.asarray(ss)
##sa = df[df.columns[1]].values
##sc = df[df.columns[2]].values
##sd = df[df.columns[3]].values
##se = df[df.columns[4]].values
##
##sb = pd.DataFrame(list(zip(ss,sa,sc,sd,se)),columns=['Path','Adenocarcinoma','Small Cell Carcinoma','Large Cell Carcinoma','Squamous Cell Carcinoma'])
##m = sb[sb.columns].values
##dataframs = []
##og_df = pd.DataFrame(columns=['Path','Adenocarcinoma','Small Cell Carcinoma','Large Cell Carcinoma','Squamous Cell Carcinoma'])
##for i in m:
##    pp = []
##    aaq = os.listdir(i[0])
##    vals = i[1]
##    val2 = i[2]
##    val3 = i[3]
##    val4 = i[4]
##
##    paths = [i[0]+'/'+j for j in aaq]
##    
##    values = [vals for j in aaq]
##    values2 = [val2 for j in aaq]
##    values3 = [val3 for j in aaq]
##    values4 = [val4 for j in aaq]
##
##    dataframs.append(pd.DataFrame(list(zip(paths,values,values2,values3,values4)),columns=['Path','Adenocarcinoma','Small Cell Carcinoma','Large Cell Carcinoma','Squamous Cell Carcinoma']))
##for i in dataframs:
##    og_df = pd.concat([og_df,i],ignore_index=True)
##print(og_df.head())

def split_by_fractions(df:pd.DataFrame, fracs:list, random_state:int=42):
    assert sum(fracs)==1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(sum(fracs))
    remain = df.index.copy().to_frame()
    res = []
    for i in range(len(fracs)):
        fractions_sum=sum(fracs[i:])
        frac = fracs[i]/fractions_sum
        idxs = remain.sample(frac=frac, random_state=random_state).index
        remain=remain.drop(idxs)
        res.append(idxs)
    return [df.loc[idxs] for idxs in res]

import numpy as np
df = pd.read_csv("train_lungpet.csv").values
og_df = pd.DataFrame(columns=['Path','Adenocarcinoma','Small Cell Carcinoma','Large Cell Carcinoma','Squamous Cell Carcinoma'])

dataframs = []
for i in df:
    uu = i[0].split('/')
    ss = uu[len(uu)-1]
    ss = ss.split('dcm')
    ss[1] = "png"
    ss = "".join(ss)
    uu[len(uu)-1]=ss
    jj = "/".join(uu)
    png_file_path = jj
    val1 = i[1]
    val2 = i[2]
    val3 = i[3]
    val4 = i[4]
    dataframs.append(pd.DataFrame([{'Path':jj,'Adenocarcinoma':val1,'Small Cell Carcinoma':val2,'Large Cell Carcinoma':val3,'Squamous Cell Carcinoma':val4}]))
    
for i in dataframs:
    og_df = pd.concat([og_df,i],ignore_index=True)

train, val = split_by_fractions(og,[0.8,0.2])
