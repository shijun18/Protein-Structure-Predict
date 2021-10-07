import os
import numpy as np
import pickle as pkl
from numpy.lib.npyio import save 
import pandas as pd
import re
from scipy.sparse import data
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


hmm_columns = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
pssm_columns = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
map_index_list = [hmm_columns.index(c) for c in pssm_columns]

amino_feats = {
    "G": [0, 22, 30, 34, -0.08, -0.88],
    "P": [1, 26, 30, 34, -0.32, -0.31],
    "T": [2, 27, 31, 34, -0.14, -0.25],
    "E": [3, 21, 29, 33, -0.7, 0.15],
    "S": [4, 27, 31, 34, -0.16, -0.45],
    "K": [5, 25, 32, 35, -0.78, 0.13],
    "C": [6, 28, 30, 34, 0.5, -0.22],
    "L": [7, 22, 30, 34, 0.76, -0.08],
    "M": [8, 28, 30, 34, 0.38, 0.18],
    "V": [9, 22, 30, 34, 0.84, -0.28],
    "D": [10, 21, 29, 33, -0.7, -0.05],
    "A": [11, 22, 30, 34, 0.36, -0.68],
    "R": [12, 25, 32, 35, -0.9, 0.53],
    "I": [13, 22, 30, 34, 0.9, -0.08],
    "N": [14, 23, 31, 34, -0.7, -0.07],
    # "H": [[15, 24, 25, 32, 33], -0.64, 0.26],
    "H": [15, 24, 25, 33, -0.64, 0.26],
    "F": [16, 24, 30, 34, 0.56, 0.40],
    "W": [17, 24, 30, 34, -0.18, 0.96],
    "Y": [18, 24, 31, 34, -0.26, 0.63],
    "Q": [19, 23, 31, 34, -0.7, 0.13],
    # "X": [[20], -0.04, 0],
    "X": [20, 20, 20, 20, -0.04, 0],
    "Z": [20, 20, 20, 20, -0.04, 0]
}


def read_hmm_file(file_path):
    '''
    filepath: pssm file
    hmm_matrix: matrix of hmm, (L,20)
    '''
    with open(file_path) as hmm_file:
        # 1. find hmm line
        line = hmm_file.readline()
        is_hmm = False
        while line:
            is_hmm = line.startswith('HMM    A')
            line = hmm_file.readline()
            if is_hmm:
                break
            else: 
                continue
        if not is_hmm:
            return None
        
        # 2. read matrix line by line
        feature_list = []
        while line:
            elmts = line.strip().split()
            if elmts and elmts[0].isalpha():
                elmts = [2**(-float(e)/1000) if e != '*' else 0. for e in elmts[2:-1]]
                feature_list.append(elmts)
            line = hmm_file.readline()
    
    # 3. exchange columns
    hmm_matrix = np.array(feature_list)
    hmm_matrix = hmm_matrix[:, map_index_list]
    return hmm_matrix 
    


def read_pssm_file(filepath):
    '''
    filepath: pssm file
    data_20:matrix of gen_pssm,(L,40)
    '''
    pssm_file = open(filepath)
    line=pssm_file.readline()
    data = []
    while line:
        items=line.strip().split()
        if items:   #不是空列表
            if items[0].isdigit():   #列表的第一个元素是数字
                data_20 = [float(items[k]) for k in range(2,22)]   #PSSM矩阵前20列-pssms_mo
                data_40 = [float(items[k])/100 for k in range(22,42)] #PSSM矩阵后20列-pssmp_mo
                data_20.extend(data_40)
                data.append(data_20)
        line=pssm_file.readline()
    pssm_file.close()
    data_arr = np.array(data)
    return data_arr

def gen_pssm_40(data):
    '''
    data: pssm matrix array
    '''
    sum = data.sum(axis = 0) #columns sum
    avg = sum / data.shape[0]
    return sum,avg

def gen_ac(data,LG,avg):
    '''
    data: pssm matrix array,(L,20)
    LG: separate parameter
    '''
    data = data-avg
    new_data = data[0:data.shape[0]-LG]*data[LG:data.shape[0]]
    sum = new_data.sum(axis = 0)/(data.shape[0]-LG)
    return sum

def gen_cc(data,LG,avg):
    data = data-avg
    new_data = np.zeros((data.shape[0]-LG,380))
    count = 0
    for i in range(20):
        for j in range(20):
            if i!=j:
                new_data[:,count] = data[0:data.shape[0]-LG,i]*data[LG:data.shape[0],j]
                count +=1
    
    sum = new_data.sum(axis = 0)/(data.shape[0]-LG)
    return sum

def gen_sxg(data,x):
    new_data = np.zeros((data.shape[0]-x-1,400))
    for i in range(20):
        for j in range(20):
            new_data[:,i*20+j] = data[0:data.shape[0]-x-1,i]*data[x+1:data.shape[0],j]
    sum = new_data.sum(axis = 0)
    return sum



def make_one_sample(pssm_path, hmm_path, id, seq, label):
    cur_dict = {}
    cur_dict['id'] = id
    cur_dict['seq'] = seq
    if label:
        cur_dict['label'] = label

    cur_dict['pssm'] = read_pssm_file(pssm_path)
    cur_dict['pssm_sum'] = cur_dict['pssm'].sum(axis=0)
    cur_dict['pssm_avg'] = cur_dict['pssm_sum'] / cur_dict['pssm'].shape[0]

    cur_dict['hmm'] = read_hmm_file(hmm_path)
    cur_dict['hmm_sum'] = cur_dict['hmm'].sum(axis=0)
    cur_dict['hmm_avg'] = cur_dict['hmm_sum'] / cur_dict['hmm'].shape[0]

    for x in range(1, 7):
        ac_pssm = gen_ac(cur_dict['pssm'][:,20:], x, cur_dict['pssm_avg'][20:])
        cc_pssm = gen_cc(cur_dict['pssm'][:,20:], x, cur_dict['pssm_avg'][20:])
        cur_dict['acc_pssm_'+str(x)] = np.concatenate([ac_pssm, cc_pssm])
        cur_dict['sxg_pssm_'+str(x - 1)] = gen_sxg(cur_dict['pssm'][:,20:], x-1)

        ac_hmm = gen_ac(cur_dict['hmm'], x, cur_dict['hmm_avg'])
        cc_hmm = gen_cc(cur_dict['hmm'], x, cur_dict['hmm_avg'])
        cur_dict['acc_hmm_'+str(x)] = np.concatenate([ac_hmm, cc_hmm])
        cur_dict['sxg_hmm_'+str(x - 1)] = gen_sxg(cur_dict['hmm'], x-1)
    
    return cur_dict

def make_vec_inputs(pssm_dir, hmm_dir, facsv_path, save_dir):
    df = pd.read_csv(facsv_path)
    ignore_list = []
    for row in tqdm(df.itertuples()):
        # print(row['key'], row['label'], row['seq'])
        if len(row) == 3:
            key, label, seq = row[1], None, row[2]
        elif len(row) == 4:
            key, label, seq = row[1], row[2], row[3]
        pssm_path = os.path.join(pssm_dir, key+'.txt')
        hmm_path = os.path.join(hmm_dir, key+'.txt')
        if not os.path.exists(pssm_path):
            ignore_list.append(pssm_path)
            continue
        sample_dict = make_one_sample(pssm_path, hmm_path, key, seq, label)
        save_path = os.path.join(save_dir, key+'.pkl')
        with open(save_path, 'wb') as pkl_file:
            pkl.dump(sample_dict, pkl_file)
    print('not found pssm files: ', ignore_list)
    return ignore_list
    

def make_one_seq_sample(pssm_path, hmm_path, id, seq, label, standard):
    cur_dict = {}
    cur_dict['id'] = id
    cur_dict['seq'] = seq
    if label:
        cur_dict['label'] = label

    pssm_mat = read_pssm_file(pssm_path) # n*40

    hmm_mat = read_hmm_file(hmm_path) # n*20
    seq_feats = np.array([amino_feats[amino] for amino in seq.upper()]) # n*6
    cur_dict['feats'] = np.concatenate([seq_feats, pssm_mat, hmm_mat], axis=1)
    if standard:
        scalar = StandardScaler()
        cur_dict['feats'] = scalar.fit_transform(cur_dict['feats'])
    return cur_dict

def make_seq_inputs(pssm_dir, hmm_dir, facsv_path, standard=False):
    df = pd.read_csv(facsv_path)
    ignore_list = []
    out_list = []
    for row in tqdm(df.itertuples()):
        if len(row) == 3: # test
            key, label, seq = row[1], None, row[2]
        elif len(row) == 4: # train
            key, label, seq = row[1], row[2], row[3]

        pssm_path = os.path.join(pssm_dir, key+'.txt')
        hmm_path = os.path.join(hmm_dir, key+'.txt')
        if not os.path.exists(pssm_path):
            ignore_list.append(pssm_path)
            continue
        sample_dict = make_one_seq_sample(pssm_path, hmm_path, key, seq, label, standard)
        out_list.append(sample_dict)
        # save_path = os.path.join(save_dir, key+'.pkl')
        # with open(save_path, 'wb') as pkl_file:
        #     pkl.dump(sample_dict, pkl_file)
    print('not found pssm files: ', ignore_list)
    return out_list

def get_list(pssm_dir, df, mode, hmm_dir):
    '''
    
    '''
    all_filenames = sorted(os.listdir(pssm_dir))
    filenames = [filename for filename in all_filenames if filename.startswith(mode)]
    output_list = []
    for filename in tqdm(filenames):
        i = int(re.findall('(\d+)',filename)[0])-1
        
        cur_dict = {}
        cur_dict['id'] =  df['key'][i]
        if mode == 'train':
            cur_dict['label'] = df['label'][i]
        cur_dict['seq'] = df['seq'][i]

        cur_dict['pssm'] = read_pssm_file(os.path.join(pssm_dir, filename))
        cur_dict['pssm_sum'], cur_dict['pssm_avg'] = gen_pssm_40(cur_dict['pssm'])
        cur_dict['hmm'] = read_hmm_file(os.path.join(hmm_dir, mode+'_'+str(i+1)+'.txt'))
        cur_dict['hmm_sum'], cur_dict['hmm_avg'] = gen_pssm_40(cur_dict['hmm'])

        for x in range(1, 7):
            ac = gen_ac(cur_dict['pssm'][:,20:], x, cur_dict['pssm_avg'][20:])
            cc = gen_cc(cur_dict['pssm'][:,20:], x, cur_dict['pssm_avg'][20:])
            cur_dict['acc_'+str(x)] = np.concatenate([ac, cc])
            cur_dict['sxg_'+str(x - 1)] = gen_sxg(cur_dict['pssm'][:,20:], x-1)

            ac_hmm = gen_ac(cur_dict['hmm'], x, cur_dict['hmm_avg'])
            cc_hmm = gen_cc(cur_dict['hmm'], x, cur_dict['hmm_avg'])
            cur_dict['acc_hmm_'+str(x)] = np.concatenate([ac_hmm, cc_hmm])
            cur_dict['sxg_hmm_'+str(x - 1)] = gen_sxg(cur_dict['hmm'], x-1)

        output_list.append(cur_dict)
    return output_list



def restore(pssm_dir, hmm_dir, save_dir):
    test_df = pd.read_csv('/staff/minfanzhao/workspace/protein-pre/dataset/test.csv')
    train_df = pd.read_csv('/staff/minfanzhao/workspace/protein-pre/dataset/train.csv')
    train_list = get_list(pssm_dir, train_df, 'train', hmm_dir)
    test_list = get_list(pssm_dir, test_df, 'test', hmm_dir)
    
    train_save_path = os.path.join(save_dir, 'train.pkl')
    test_save_path = os.path.join(save_dir, 'test.pkl')
    
    with open(train_save_path, 'wb') as train_file:
        pkl.dump(train_list, train_file)

    with open(test_save_path, 'wb') as test_file:
        pkl.dump(test_list, test_file)

if __name__ == '__main__':
    ''' save as one list'''
    # hmm_dir = '/staff/wangzhaohui/proteinFold/data/hmm_uniclust30_2018_08'
    # pssm_dir = '/staff/minfanzhao/blast/pssm/'
    # save_dir = '/staff/wangzhaohui/proteinFold/data/combined_data'
    # restore(pssm_dir, hmm_dir, save_dir)

    ''' each sample is corresponding to a file'''
    
    # mode = 'train'
    # facsv_path = '/staff/wangzhaohui/proteinFold/protein_predict/dataset/' + mode + '.csv'
    # hmm_dir = '/staff/wangzhaohui/proteinFold/data/feature_raw/hmm_uniclust30_2018_08_idname/' + mode
    # pssm_dir = '/staff/wangzhaohui/proteinFold/data/feature_raw/pssm_idname/' + mode
    # save_dir = '/staff/wangzhaohui/proteinFold/data/model_inputs/split_input_hmmclust_pssmblast/' + mode
    # make_vec_inputs(pssm_dir, hmm_dir, facsv_path, save_dir)

    ''' transformer inputs'''
    mode = 'train'
    facsv_path = '/staff/wangzhaohui/proteinFold/protein_predict/dataset/' + mode + '.csv'
    hmm_dir = '/staff/wangzhaohui/proteinFold/data/feature_raw/hmm_uniclust30_2018_08_idname/' + mode
    pssm_dir = '/staff/wangzhaohui/proteinFold/data/feature_raw/pssm_idname/' + mode
    # save_dir = '/staff/wangzhaohui/proteinFold/data/model_inputs/seq_inputs/hmmclust_pssmblast/' + mode
    save_path = '/staff/wangzhaohui/proteinFold/data/model_inputs/seq_inputs/standard_hmmclust_pssmblast_list/' + mode + '.pkl'
    sample_list = make_seq_inputs(pssm_dir, hmm_dir, facsv_path, standard=True)
    print(len(sample_list))
    with open(save_path, 'wb') as save_file:
        pkl.dump(sample_list, save_file)


    # test case
    # sample_path = '/staff/wangzhaohui/proteinFold/data/model_inputs/seq_inputs/hmmclust_pssmblast/test/d3cs0a2.pkl'

    with open(save_path, 'rb') as pklfile:
        data_list = pkl.load(pklfile)
    # for sample in data:
    #     if sample['id'] == 'd1skva1':
    #         print(sample)
    sample_index = 23
    sample = data_list[sample_index]
    print(sample.keys())
    print(sample['feats'].shape)
    print(sample['seq'], len(sample['seq']))
    # # print(sample['label'])

    # print(sample['id'])

