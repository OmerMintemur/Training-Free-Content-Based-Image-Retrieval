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
import ast

def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# Read the database
database = pd.read_csv('Database.csv',converters={'R_Prob_Data': ast.literal_eval,
                                                  'G_Prob_Data': ast.literal_eval,
                                                  'B_Prob_Data': ast.literal_eval})
mapping = {'beaches':0,
           'bus':1,
           'dinosaurs':2,
           'elephants':3,
           'flowers':4,
           'foods':5,
           'horses':6,
           'monuments':7,
           'mountains_and_snow':8,
           'people_and_villages_in_Africa':9}
path = "dataset\\test_set\\"
# Read a random image from the 'test set'
# Get a random class
classes = os.listdir(path)
selected_class = random.choice(classes)

# Select a random image from random class
image = random.choice(glob.glob(path+selected_class+'\\*.jpg'))
query_image = cv2.imread(image)
query_image = cv2.resize(query_image,(128,128))
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

# Calculate necessary statistics (Dominant Colors, Probs for KL)
# KL
hist = [np.histogram(query_image[:,:,x], bins=255)[0] for x in range(3)]
hist = [gaussian_filter(hist[x], 1) for x in range(3)]
hist = [hist[x]+1 for x in range(3)]

RProb_Q = [1 + (float(i)/sum(hist[0])) for i in hist[0]]
GProb_Q = [1 + (float(i)/sum(hist[1])) for i in hist[1]]
BProb_Q = [1 + (float(i)/sum(hist[2])) for i in hist[2]]

# Dominant Colors
dominant_color = ColorThief(image).get_color(quality=1)
Dominant_R_Color_Q = dominant_color[0]
Dominant_G_Color_Q = dominant_color[1]
Dominant_B_Color_Q = dominant_color[2]
# The most dominant for query
Most_Dominant_Query = max([Dominant_R_Color_Q,Dominant_G_Color_Q,Dominant_B_Color_Q])


# Start Traversing
# Save Image Class, Path, Result
Image_Class = []
Image_Path =  []
Result =      []
Dominant =    []
for index, row in database.iterrows():
    # Dominant Colors for Database
    D_R_Color = row['Dominant_R_Color']
    D_G_Color = row['Dominant_G_Color']
    D_B_Color = row['Dominant_B_Color']
    # Get dominant
    Most_Dominant = max([D_R_Color,D_G_Color,D_B_Color])
    # Probs
    R_Prob_Data = row['R_Prob_Data'][0]
    G_Prob_Data = row['G_Prob_Data'][0]
    B_Prob_Data = row['B_Prob_Data'][0]

    

 
    # Read a database image
    d_image = cv2.resize(cv2.imread(row['Image_Path']),(128,128))
    s = ssim(d_image, query_image, multichannel=True,channel_axis=-1)

    # KL divergence
    kl = (kl_divergence(R_Prob_Data,RProb_Q) + kl_divergence(G_Prob_Data,GProb_Q) + kl_divergence(B_Prob_Data,BProb_Q))

    if Most_Dominant == Most_Dominant_Query:
        result = (s/kl) * 2
    else:
        result = (s/kl) / 2
    """dominant = np.array([D_R_Color, D_G_Color, D_B_Color]) - np.array([Dominant_R_Color_Q, Dominant_G_Color_Q, Dominant_B_Color_Q])
    if dominant.sum() == 0:
        result = s/(kl+abs(dominant.sum())) * 2
    else:
        result = s/(kl+abs(dominant.sum())) / 2"""
    
    Image_Class.append(row['Image_Class'])
    Image_Path.append(row['Image_Path'])
    Result.append(result)
    # Dominant.append(dominant)

dict = {'Image_Path':  Image_Path,
        'Image_Class': Image_Class,
        'Result': Result} 
df = pd.DataFrame(dict)

results = df.sort_values(['Result'],ascending = [False])


k=5
# Plot top-k
first_k_image = results.iloc[:k,:]
fig, axes = plt.subplots(1, k, figsize=(18, 3))  # Adjust figure size

# Loop through axes and display images
for i, ax in enumerate(axes):
    im = cv2.resize(cv2.imread(first_k_image.iloc[i,0]),(128,128))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax.imshow(im)
    ax.axis('off')  # Hide axes

plt.tight_layout()  # Adjust layout
plt.show()

# Calculate precision
selected_class_index = mapping[selected_class]
print(f'Query image is {selected_class}, index is {selected_class_index}')

# Map the output
first_k_image = first_k_image['Image_Class'].map(mapping)
retrieved_classes = first_k_image.to_list()
print(f'Retrieved Classes are {retrieved_classes}')

def average_precision(relevant, retrieved):
    relevant_count = 0
    precision_sum = 0

    for i, img in enumerate(retrieved):
        if img in relevant:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precision_sum += precision

    return precision_sum / len(retrieved) if relevant else 0
ap_score = average_precision([selected_class_index], retrieved_classes)
print(f"Average Precision (AP): {ap_score:.4f}")
