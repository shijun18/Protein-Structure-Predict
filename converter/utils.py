import pandas as pd

def csv_reader_single(csv_file,key_col=None,value_col=None):
    '''
    Extracts the specified single column, return a single level dict.
    The value of specified column as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique. 
    - value_col: string,  specified column as value
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = file_csv[value_col].values.tolist()
    
    target_dict = {}
    for key_item,value_item in zip(key_list,value_list):
        target_dict[key_item] = value_item

    return target_dict

if __name__ == '__main__':
    # data_path = '../dataset/train.csv'
    # type_dict = csv_reader_single(data_path,'label','key')
    # print(len(set(list(type_dict.keys()))))

    data_path = '../dataset/test.csv'
    test_csv = pd.read_csv(data_path)
    test_keys = test_csv['key'].values.tolist()
    test_dict = csv_reader_single(data_path,'key','seq')

    result_path = '../dataset/result.csv'
    result_dict = csv_reader_single(result_path,'key','label')

    reverse_result_dict = csv_reader_single(result_path,'seq','label')

    test_result = []
    for key in test_keys:
        item = [key]
        try:
            item.append(result_dict[key])
        except:
            item.append(reverse_result_dict[test_dict[key]])
        test_result.append(item)
    col = ['sample_id','category_id']
    csv_df = pd.DataFrame(data=test_result,columns=col)
    csv_df.to_csv('./test_result.csv',index=False)
