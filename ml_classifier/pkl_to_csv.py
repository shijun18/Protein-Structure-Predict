import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

def pkl_to_csv(pkl_file,csv_path,exclude_key=['pssm','hmm']):

    csv_data = {
        'id':[],
        'seq':[]
    }
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)

    for item in tqdm(data):
        for key in item.keys():
            if key not in exclude_key:
                # print(key)
                if isinstance(item[key],str):
                    if key not in csv_data.keys():
                        csv_data[key] = []
                    csv_data[key].append(item[key])
                    
                else:
                    assert len(item[key].shape) == 1
                    fea_len = item[key].shape[0]
                    for i in range(fea_len):
                        if f'{key}_{str(i)}' not in csv_data.keys():
                            csv_data[f'{key}_{str(i)}'] = []
                        csv_data[f'{key}_{str(i)}'].append(item[key][i])
                    # print(item[key].shape)
    print(len(csv_data.keys()))
    csv_file = pd.DataFrame(data=csv_data)
    csv_file.to_csv(csv_path, index=False)




if __name__ == '__main__':

    hmm_key = ['hmm_sum', 'hmm_avg','acc_hmm_1','sxg_hmm_0','acc_hmm_2','sxg_hmm_1','acc_hmm_3','sxg_hmm_2','acc_hmm_4','sxg_hmm_3','acc_hmm_5','sxg_hmm_4','acc_hmm_6','sxg_hmm_5']
    pssm_key = ['pssm_sum', 'pssm_avg','acc_1','sxg_0','acc_2','sxg_1','acc_3','sxg_2','acc_4','sxg_3','acc_5','sxg_4','acc_6','sxg_5']

    train_pkl = '/staff/minfanzhao/workspace/protein_predict/dataset/train.pkl'
    train_csv = '../dataset/pssm_train.csv'
    pkl_to_csv(train_pkl,train_csv,['pssm','hmm'] + hmm_key)

    test_pkl = '/staff/minfanzhao/workspace/protein_predict/dataset/test.pkl'
    test_csv = '../dataset/pssm_test.csv'
    pkl_to_csv(test_pkl,test_csv,['pssm','hmm'] + hmm_key)