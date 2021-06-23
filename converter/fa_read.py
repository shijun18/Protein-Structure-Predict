import pandas as pd



def convert_train(data_path,save_path):

    f = open(data_path,'r')
    seq_dict = {
        'key':[],
        'label':[],
        'seq':[]
    }

    count = 0
    for line in f:
        line = line.strip('\n')
        if line.startswith('>'):
            count += 1
            key = line.split(' ')[0][1:]
            label = line.split(' ')[1].split('.')
            label = label[0] + '.' + label[1]
            seq = ''
            
            seq_dict['key'].append(key)
            seq_dict['label'].append(label)
            seq_dict['seq'].append(seq)
            
        else:
            seq_dict['seq'][count-1] += line


    train_df = pd.DataFrame(data=seq_dict)
    train_df.to_csv(save_path,index=False)



def convert_test(data_path,save_path):

    f = open(data_path,'r')

    seq_dict = {
        'key':[],
        'seq':[]
    }
    count = 0
    for line in f:
        line = line.strip('\n')
        if line.startswith('>'):
            count +=1
            key = line.rstrip()[1:]
            seq = ''
            seq_dict['key'].append(key)
            seq_dict['seq'].append(seq)
            
        else:
            seq_dict['seq'][count-1] += line
        
    train_df = pd.DataFrame(data=seq_dict)
    train_df.to_csv(save_path,index=False)


if __name__ == '__main__':
    # data_path = '../dataset/astral_train.fa'
    # save_path = '../dataset/train.csv'
    # convert_train(data_path,save_path)

    # data_path = '../dataset/astral_test.fa'
    # save_path = '../dataset/test.csv'
    # convert_test(data_path,save_path)
    data_path = '../dataset/result.fa'
    save_path = '../dataset/result.csv'
    convert_train(data_path,save_path)
