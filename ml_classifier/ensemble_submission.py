import os
import glob
import pandas as pd
import numpy as np


def vote_ensemble(csv_path_list, save_path, col='category_id'):
    result = {}
    ensemble_list = []
    for csv_path in csv_path_list:
        csv_file = pd.read_csv(csv_path)
        ensemble_list.append(csv_file[col].values.tolist())

    result['sample_id'] = csv_file['sample_id'].values.tolist()
    vote_array = np.array(ensemble_list)
    result['category_id'] = [
        max(list(vote_array[:, i]), key=list(vote_array[:, i]).count)
        for i in range(vote_array.shape[1])
    ]

    final_csv = pd.DataFrame(result)
    final_csv.to_csv(save_path, index=False)



def find_csv(csv_dir,prefix='fusion'):
    csv_path = []
    for trial in os.scandir(csv_dir):
        # print(trial.path)
        if trial.is_dir():
            try:
                csv_file = glob.glob(os.path.join(trial.path,prefix + '-prob.csv'))[0]
                csv_path.append(csv_file)
            except:
                continue
    
    return csv_path




if __name__ == "__main__":

    save_path = './result/fusion.csv'

    csv_dir = ['./result/hmm_qtr_scale_pro/','./result/hmm_half_scale_pro/']
    csv_path_list = []
    for item in csv_dir:
        csv_path_list += find_csv(item)
    print('Ensemble Len:',len(csv_path_list))
    vote_ensemble(csv_path_list, save_path)
    
