import csv
import pandas as pd 
import json 

dataset= 'adult'
save_dir = 'results_part_4'
fileName = '%s/results_4_1_learn_post_black_boxes_%s.csv' %(save_dir, dataset)

results_df = pd.read_csv(fileName)
results_dict = {}
for index, row in results_df.iterrows():
    #print(index, row)
    row_trails_str = row['Trials Details']
    print(row_trails_str)
    row_trials_list = (row_trails_str)
    
    # json.loads(row_trails_str)
    print(row_trials_list)
    exit()