import os
import glob
import pandas as pd
import numpy as np
from pandas import tseries
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,f1_score
from utils import csv_reader_single

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



def eval_metric(result_dict,test_id,pred_result):
    le = LabelEncoder()
    true_result = [result_dict[case] for case in test_id]
    true_result = le.fit_transform(true_result)
    pred_result = le.transform(pred_result)
    acc = accuracy_score(true_result,pred_result)
    f1 = f1_score(true_result,pred_result,average='macro')

    print('Evaluation:')
    print('Accuracy:%.5f'%acc)
    print('F1 Score:%.5f'%f1)

    return acc,f1


def find_csv(csv_dir,prefix='fusion'):
    csv_path = []
    for trial in os.scandir(csv_dir):
        # print(trial.path)
        if trial.is_dir():
            try:
                csv_file = glob.glob(os.path.join(trial.path,prefix + '*.csv'))[0]
                if eval(os.path.splitext(os.path.basename(csv_file))[0].split('-')[2]) > 0.78:
                    csv_path.append(csv_file)
            except:
                continue
    
    return csv_path




if __name__ == "__main__":

    save_path = './result/fusion.csv'
    # csv_dir = ['./result/pssm_qtr_scale/','./result/pssm_half_scale/','./result/pssm_uncia_scale/']
    csv_dir = ['./result/pssm_qtr_scale/','./result/pssm_half_scale/','./result/pssm_half_scale_aug_rs_tta/']
    # csv_dir = ['./result/pssm_half_scale_aug_rs_tta/']
    result_path = '../converter/test_result.csv'
    csv_path_list = []
    for item in csv_dir:
        csv_path_list += find_csv(item)
    print('Ensemble Len:',len(csv_path_list))
    vote_ensemble(csv_path_list, save_path)
    result_dict = csv_reader_single(result_path,'sample_id','category_id')  
    fusion_df = pd.read_csv(save_path)
    test_id = fusion_df['sample_id'].values.tolist()
    pred_result = fusion_df['category_id'].values.tolist()
    eval_metric(result_dict,test_id,pred_result)
    
