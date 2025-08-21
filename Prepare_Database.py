import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import log2
from scipy.ndimage import gaussian_filter
import glob
import pandas as pd
from colorthief import ColorThief
import random
import os



# Path for the all set
path = "dataset\\training_set\\"

# Read folder in 'path'
classes = os.listdir(path)

Image_Path =  []
R_Prob_Data = []
G_Prob_Data = []
B_Prob_Data = []
Image_Class = []
Dominant_R_Color = []
Dominant_G_Color = []
Dominant_B_Color = []
# Read each image according to 'classes'
for c in classes:
    for image in glob.glob(f'{path}{c}/*.jpg'):
        im = cv2.imread(image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im,(128,128))
        # Save Image Path and Class
        Image_Path.append(str(image))
        Image_Class.append(str(c))

        # Get R_Prob_Data, G_Prob_Data, B_Prob_Data
        hist = [np.histogram(im[:,:,x], bins=255)[0] for x in range(3)]
        hist = [gaussian_filter(hist[x], 1) for x in range(3)]
        hist = [hist[x]+1 for x in range(3)]

        RProb_Q = [1 + (float(i)/sum(hist[0])) for i in hist[0]]
        GProb_Q = [1 + (float(i)/sum(hist[1])) for i in hist[1]]
        BProb_Q = [1 + (float(i)/sum(hist[2])) for i in hist[2]]

        # Save R_Prob_Data, G_Prob_Data, B_Prob_Data
        R_Prob_Data.append([RProb_Q])
        G_Prob_Data.append([GProb_Q])
        B_Prob_Data.append([BProb_Q])

        # Get dominant colors
        dominant_color = ColorThief(image).get_color(quality=1)
        Dominant_R_Color.append(dominant_color[0])
        Dominant_G_Color.append(dominant_color[1])
        Dominant_B_Color.append(dominant_color[2])
    print(f'{c} is done')

dict = {'Image_Path':  Image_Path,
        'Image_Class': Image_Class,
        'R_Prob_Data': R_Prob_Data,
        'G_Prob_Data': G_Prob_Data,
        'B_Prob_Data': B_Prob_Data,
        'Dominant_R_Color': Dominant_R_Color,
        'Dominant_G_Color': Dominant_G_Color,
        'Dominant_B_Color': Dominant_B_Color} 
df = pd.DataFrame(dict)
print(df)
df.to_csv("Database.csv")
