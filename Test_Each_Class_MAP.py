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
import pickle

def ret_matrix(retrieval_matrix,results,current_index):
    first_k_image = results.iloc[:K,:]
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
    # Calculate precision
    selected_class_index = mapping[folder]
    # print(f'Query image is {folder}, index is {selected_class_index}')

    # Map the output
    first_k_image = first_k_image['Image_Class'].map(mapping)
    retrieved_classes = first_k_image.to_list()
    # print(retrieved_classes)
    retrieval_matrix[current_index,retrieved_classes]+=1

    return retrieval_matrix

def precision(results,K,folder):
    first_k_image = results.iloc[:K,:]
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
    # Calculate precision
    selected_class_index = mapping[folder]
    # print(f'Query image is {folder}, index is {selected_class_index}')

    # Map the output
    first_k_image = first_k_image['Image_Class'].map(mapping)
    retrieved_classes = first_k_image.to_list()
    # print(f'Retrieved Classes are {retrieved_classes}')


    relevant = [selected_class_index]
    retrieved = retrieved_classes
    relevant_count = 0
    precision_sum = 0

    for i, img in enumerate(retrieved):
        if img in relevant:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precision_sum += precision

    return precision_sum / len(retrieved) if relevant else 0
        

def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))


# Read the database
database = pd.read_csv('Database.csv',converters={'R_Prob_Data': ast.literal_eval,
                                                  'G_Prob_Data': ast.literal_eval,
                                                  'B_Prob_Data': ast.literal_eval})
folders = ['beaches','bus','dinosaurs','elephants','flowers',
           'foods','horses','monuments','mountains_and_snow',
           'people_and_villages_in_Africa']



overall_K_results = {}
# For each image in the test folder
# We traverse and calculate MAP for the folder
# This is the main For Loop
path = "dataset\\test_set\\"
for K in [1,5,7,9,20,50,100]: # Top-K results
    retrieval_matrix = np.zeros((10,10))
    current_index = 0
    dict_for_mean_average_precision={x:0 for x in folders}
    for folder in folders:
        im_name=""
        avg_precision = 0.0
        for image in glob.glob(path+folder+'\\*.jpg'):
            im_name=image
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
            Most_Dominant_Query = [Dominant_R_Color_Q,Dominant_G_Color_Q,Dominant_B_Color_Q]

            # Start traversing the database
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
                Most_Dominant = [D_R_Color,D_G_Color,D_B_Color]
                # Probs
                R_Prob_Data = row['R_Prob_Data'][0]
                G_Prob_Data = row['G_Prob_Data'][0]
                B_Prob_Data = row['B_Prob_Data'][0]
            
                # Read a database image
                d_image = cv2.resize(cv2.imread(row['Image_Path']),(128,128))
                s = ssim(d_image, query_image, multichannel=True,channel_axis=-1)

                # KL divergence
                kl = (kl_divergence(R_Prob_Data,RProb_Q) + kl_divergence(G_Prob_Data,GProb_Q) + kl_divergence(B_Prob_Data,BProb_Q))

                color_mean = abs(np.subtract(Most_Dominant, Most_Dominant_Query))
                result = s / (kl+(np.mean(color_mean)))

                """if Most_Dominant == Most_Dominant_Query:
                    result = (s/kl) * 2
                else:
                    result = (s/kl) / 2"""

                Image_Class.append(row['Image_Class'])
                Image_Path.append(row['Image_Path'])
                Result.append(result) 

            dict = {'Image_Path': Image_Path, 'Image_Class': Image_Class, 'Result': Result} 
            df = pd.DataFrame(dict)

            results = df.sort_values(['Result'],ascending = [False])    
            # print(f'Precision for the class {folder} - Image is {im_name} - {precision(results,K,folder)}')
            dict_for_mean_average_precision[folder]+=precision(results,K,folder) / len(glob.glob(path+folder+'\\*.jpg'))
            retrieval_matrix = ret_matrix(retrieval_matrix,results,current_index)
        current_index+=1
    overall_K_results[K] = dict_for_mean_average_precision
    print(f"K {K} is done") 
    # Normalize along dim=1 (row-wise)
    row_sums = retrieval_matrix.sum(axis=1, keepdims=True)  # Sum for each row
    retrieval_matrix = np.divide(retrieval_matrix, row_sums, where=row_sums != 0)  # Avoid division by zero
    print(retrieval_matrix) 
    plt.imshow(retrieval_matrix)
    plt.show()
    np.save(f'Correlation_Matrix_K_{K}.npy', retrieval_matrix)
    

    
    
with open('Overall_K_Results.pkl', 'wb') as f:
    pickle.dump(overall_K_results, f)
    