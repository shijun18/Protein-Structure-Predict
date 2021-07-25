import pandas as pd


AMAC = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']

def statistic_fea(input_csv,save_csv,stride_list=[1,2,3,4]):
    df = pd.read_csv(input_csv)
    for amac in AMAC:
        for stride in stride_list:
            df[f'{amac}_{stride}_num'] = df['seq'].apply(lambda x: str(x).lower()[::stride].count(amac)/ len(str(x)[::stride]))
    df.to_csv(save_csv,index=False)


if __name__ == '__main__':
    train_csv = '../dataset/train.csv'
    save_csv = '../dataset/fea_train.csv'

    statistic_fea(train_csv,save_csv)

    test_csv = '../dataset/test.csv'
    save_csv = '../dataset/fea_test.csv'
    statistic_fea(test_csv,save_csv)

    result_csv = '../dataset/result.csv'
    save_csv = '../dataset/fea_result.csv'
    statistic_fea(result_csv,save_csv)
