import os
import shutil
import pickle
from torch import nn
import torch
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from utils import dfs_remove_weight
from feature_selection import select_feature_linesvc


class MyDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        data = self.X[index]
        if self.Y is not None:
            target = self.Y[index]
            sample = {'data':torch.from_numpy(data), 'target':int(target)}
        else:
            sample = {'data':torch.from_numpy(data)}
        
        return sample


class MLP_CLASSIFIER(nn.Module):
    def __init__(self, input_size, output_size=245, depth=3, depth_list=[256,128,64], drop_prob=0.5, use_norm=True):
        super(MLP_CLASSIFIER, self).__init__()

        assert len(depth_list) == depth
        self.linear_list = []
        for i in range(depth):
            if i == 0:
                self.linear_list.append(nn.Linear(input_size,depth_list[i]))
            else:
                self.linear_list.append(nn.Linear(depth_list[i-1],depth_list[i]))
            if use_norm:
                self.linear_list.append(nn.BatchNorm1d(depth_list[i]))
            self.linear_list.append(nn.ReLU(inplace=True))
            # self.linear_list.append(nn.Tanh())
            # self.linear_list.append(nn.Dropout(0.2))
            

        self.linear = nn.Sequential(*self.linear_list)
        self.drop = nn.Dropout(drop_prob) if drop_prob > 0.0 else None
        self.cls_head = nn.Linear(depth_list[-1],output_size)

    def forward(self, x):
        x = self.linear(x) #N*C
        if self.drop:
            x = self.drop(x)
        x = self.cls_head(x)
        return x


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''
    Computes the precision@k for the specified values of k
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res


def train_epoch(epoch,net,criterion,optim,train_loader,scaler,use_fp16=True):
    net.train()
   
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for step, sample in enumerate(train_loader):

        data = sample['data']
        target = sample['target']

        # print(data.size())
        data = data.cuda()
        target = target.cuda()
        with autocast(use_fp16):
            output = net(data)
            loss = criterion(output, target)
        
        optim.zero_grad()
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc = accuracy(output.data, target)[0]
        train_loss.update(loss.item(), data.size(0))
        train_acc.update(acc.item(), data.size(0))

        torch.cuda.empty_cache()

        # if step % 10 == 0:
        #     print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
        #             .format(epoch, step, loss.item(), acc.item(), optim.param_groups[0]['lr']))

    return train_acc.avg,train_loss.avg


def val_epoch(epoch,net,criterion,val_loader,use_fp16=True):
    
    net.eval()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for step, sample in enumerate(val_loader):
            data = sample['data']
            target = sample['target']

            data = data.cuda()
            target = target.cuda()
            with autocast(use_fp16):
                output = net(data)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            val_loss.update(loss.item(), data.size(0))
            val_acc.update(acc.item(), data.size(0))

            torch.cuda.empty_cache()

            # print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'
            #     .format(epoch, step, loss.item(), acc.item()))

    return val_acc.avg,val_loss.avg


def evaluation(test_data,net,weight_path,use_fp16=True):
    ckpt = torch.load(weight_path)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    test_data = torch.from_numpy(test_data) #N*fea_len
    with torch.no_grad():
        data = test_data.cuda()
        with autocast(use_fp16):
            output = net(data)
        output = output.float()
        output = torch.softmax(output,dim=1)
        output = torch.argmax(output,dim=1) #N
        torch.cuda.empty_cache()

    return output.cpu().numpy().tolist()


def eval_metric(result_dict,test_id,pred_result,le):
    true_result = [result_dict[case] for case in test_id]
    true_result = le.transform(true_result)
    acc = accuracy_score(true_result,pred_result)
    f1 = f1_score(true_result,pred_result,average='macro')

    print('Evaluation:\n')
    print('Accuracy:%.5f'%acc)
    print('F1 Score:%.5f'%f1)

    return acc,f1


def manual_select(total_list,exclude_list=None):
    fea_list = []
    nouse_list = []
    if exclude_list is not None:
        for exclude_label in exclude_list:
            if 'sum' not in exclude_label and 'avg' not in exclude_label:
                nouse_list += [f'{exclude_label}_{str(i)}' for i in range(400)]
            else:
                nouse_list += [f'{exclude_label}_{str(i)}' for i in range(40)]

        for col in total_list:
            if col not in nouse_list:
                fea_list.append(col)
    
        return fea_list
    else:
        return total_list



def run(train_path,test_path,result_path,output_dir,net_depth=3,exclude_list=None,scale_flag=True,select_flag=False):

    torch.manual_seed(0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    # load training and testing data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # result
    from utils import csv_reader_single
    result_dict = csv_reader_single(result_path,'sample_id','category_id')    

    # data preprocessing
    del train_df['seq']
    del test_df['seq']

    test_id = test_df['id']
    # print(list(test_id))
    del train_df['id']
    del test_df['id']

    # manual select
    fea_list = manual_select(train_df.columns,exclude_list)
    fea_list = [f for f in fea_list if f not in ['label']]

    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])
    num_classes = len(set(train_df['label']))
    fea_len = len(fea_list)

    # convert to numpy array
    Y = np.asarray(train_df['label']).astype(np.uint8)
    X = np.asarray(train_df[fea_list]).astype(np.float32)
    test = np.asarray(test_df[fea_list]).astype(np.float32)
    
    # feature selection
    if select_flag:
        select_model_path = './select_model.pkl'
        if os.path.exists(select_model_path):
            with open(select_model_path, 'rb') as f:
                select_model = pickle.load(f)
        else:
            select_model = select_feature_linesvc(X, Y, select_model_path)

        X = select_model.transform(X)
        test = select_model.transform(test)

    # data scale
    if scale_flag:
        X_len = X.shape[0]
        data_scaler = StandardScaler()
        cat_data = np.concatenate([X,test],axis=0)
        cat_data= data_scaler.fit_transform(cat_data)

        X = cat_data[:X_len]
        test = cat_data[X_len:]
    
    # print(Y)
    print(X.shape,test.shape)


    total_result = []
    kfold = KFold(n_splits=5,shuffle=True,random_state=21)
    for fold_num,(train_index,val_index) in enumerate(kfold.split(X)):
        print(f'***********fold {fold_num+1} start!!***********')
        fold_dir = os.path.join(output_dir,f'fold{fold_num+1}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        # initialization
        epoch_num = 100
        acc_threshold = 0.0

        depth = net_depth
        depth_list = [int(fea_len*(2**(1-i))) for i in range(depth)]
        
        net = MLP_CLASSIFIER(fea_len,num_classes,depth,depth_list)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [25,60,80], gamma=0.1)
        scaler = GradScaler()
        
        net = net.cuda()
        criterion = criterion.cuda()

        # data loader
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        print('Train Data Size:',x_train.shape)
        print('Val Data Size:',x_val.shape)

        train_dataset = MyDataset(X=x_train,Y=y_train)
        val_dataset = MyDataset(X=x_val,Y=y_val)

        train_loader = DataLoader(
                train_dataset,
                batch_size=256,
                shuffle=True,
                num_workers=2)
        
        val_loader = DataLoader(
                val_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=2)

        # main processing
        for epoch in range(epoch_num):
            train_acc, train_loss = train_epoch(epoch,net,criterion,optim,train_loader,scaler)
            val_acc,val_loss = val_epoch(epoch,net,criterion,val_loader)
            torch.cuda.empty_cache()

            if epoch % 10 == 0:
                print('Train epoch:{},train_loss:{:.5f},train_acc:{:.5f}'
                    .format(epoch, train_loss, train_acc))

                print('Val epoch:{},val_loss:{:.5f},val_acc:{:.5f}'
                    .format(epoch, val_loss, val_acc))


            if lr_scheduler is not None:
                lr_scheduler.step()
            
            if val_acc > acc_threshold:
                acc_threshold = val_acc
                saver = {
                    'state_dict': net.state_dict()
                }

                file_name = 'epoch:{}-val_acc:{:.5f}-val_loss:{:.5f}-mlp.pth'.format(epoch,val_acc,val_loss)
                save_path = os.path.join(fold_dir, file_name)
                print('Save as: %s'%file_name)
                torch.save(saver, save_path)
        
        # save top3 model
        dfs_remove_weight(fold_dir,retain=3)

        # generating test result using the best model
        fold_result = evaluation(test,net,save_path)
        acc,f1 = eval_metric(result_dict,test_id,fold_result,le)
        total_result.append(fold_result)
        fold_result = le.inverse_transform(fold_result)
        
        # csv save
        fold_csv = {}
        fold_csv = pd.DataFrame(fold_csv)
        fold_csv['sample_id'] = list(test_id) + ['d2ciob_']
        fold_csv['category_id'] = list(fold_result) + ['b.1']
        fold_csv.to_csv(os.path.join(output_dir, f'fold{fold_num}_acc-{round(acc,4)}_f1-{round(f1,4)}.csv'),index=False)

    # result fusion by voting
    final_result = []
    vote_array = np.asarray(total_result).astype(np.uint8)
    final_result.extend([max(list(vote_array[:,i]),key=list(vote_array[:,i]).count) for i in range(vote_array.shape[1])])
    acc,f1 = eval_metric(result_dict,test_id,final_result,le)
    final_result = le.inverse_transform(final_result)

    # csv save
    total_csv = {}
    total_csv = pd.DataFrame(total_csv)
    total_csv['sample_id'] = list(test_id) + ['d2ciob_']
    total_csv['category_id'] = list(final_result) + ['b.1']
    total_csv.to_csv(os.path.join(output_dir, f'fusion_acc-{round(acc,4)}_f1-{round(f1,4)}.csv'),index=False)


if __name__ == '__main__':
    pssm_key = ['pssm_sum', 'pssm_avg','acc_1','acc_2','acc_3','acc_4','acc_5','acc_6','sxg_0','sxg_1','sxg_2','sxg_3','sxg_4','sxg_5']
    hmm_key = ['hmm_sum', 'hmm_avg','acc_hmm_1','acc_hmm_2','acc_hmm_3','acc_hmm_4','acc_hmm_5','acc_hmm_6','sxg_hmm_0','sxg_hmm_1','sxg_hmm_2','sxg_hmm_3','sxg_hmm_4','sxg_hmm_5']
    
    train_path = '../dataset/pssm_train.csv'
    test_path = '../dataset/pssm_test.csv'
    # train_path = '../dataset/hmm_train.csv'
    # test_path = '../dataset/hmm_test.csv'
    result_path = '../converter/test_result.csv'
    output_dir = './pssm_total_scale/'

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    for i in range(100):
        save_path = os.path.join(output_dir,f'trial_{str(i+1)}')
        # exclude_list = random.sample(pssm_key[2:],6)
        exclude_list = None
        # exclude_list = random.sample(hmm_key[2:],6)
        print('exclude list:',exclude_list)
        run(train_path,test_path,result_path,save_path,net_depth=3,exclude_list=exclude_list,scale_flag=True,select_flag=False)
