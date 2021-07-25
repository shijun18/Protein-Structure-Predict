import os
from torch import nn
import torch
import pandas as pd

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,f1_score
from utils import dfs_remove_weight

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
    def __init__(self, input_size, output_size=245, depth=3, depth_list=[256,128,64], drop_prob=0.5):
        super(MLP_CLASSIFIER, self).__init__()

        assert len(depth_list) == depth
        self.linear_list = []
        for i in range(depth):
            if i == 0:
                self.linear_list.append(nn.Linear(input_size,depth_list[i]))
            else:
                self.linear_list.append(nn.Linear(depth_list[i-1],depth_list[i]))
            self.linear_list.append(nn.ReLU(depth_list[i]))
            # self.linear_list.append(nn.Tanh())

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

        if step % 200 == 0:
            print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
                    .format(epoch, step, loss.item(), acc.item(), optim.param_groups[0]['lr']))

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


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    train_path = '../dataset/pssm_train.csv'
    train_df = pd.read_csv(train_path)


    test_path = '../dataset/pssm_test.csv'
    test_df = pd.read_csv(test_path)

    result_path = '../converter/test_result.csv'
    from utils import csv_reader_single
    result_dict = csv_reader_single(result_path,'sample_id','category_id')    

    output_dir = './model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    del train_df['seq']
    del test_df['seq']

    test_id = test_df['id']
    del train_df['id']
    del test_df['id']

    fea_list = [f for f in train_df.columns if f not in ['label']] 

    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])
    num_classes = len(set(train_df['label']))
    fea_len = len(fea_list)

    Y = np.asarray(train_df['label']).astype(np.uint8)
    X = np.asarray(train_df[fea_list]).astype(np.float32)
    test = np.asarray(test_df[fea_list]).astype(np.float32)

    kfold = KFold(n_splits=5,shuffle=True,random_state=21)

    # print(Y)
    print(X.shape,test.shape)

    total_result = []

    depth = 3
    depth_list = [int(fea_len*(2**(1-i))) for i in range(depth)]

    for fold_num,(train_index,val_index) in enumerate(kfold.split(X)):
        print(f'***********fold {fold_num+1} start!!***********')
        epoch_num = 100
        acc_threshold = 0.0

        fold_dir = os.path.join(output_dir,f'fold{fold_num+1}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        net = MLP_CLASSIFIER(fea_len,num_classes,depth,depth_list)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
        # optim = torch.optim.Adam(net.parameters(),lr=0.05,weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [25,60,80], gamma=0.1)
        # lr_scheduler = None
        scaler = GradScaler()
        
        net = net.cuda()
        criterion = criterion.cuda()

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
        
        fold_csv = {}
        fold_csv = pd.DataFrame(fold_csv)
        fold_csv['sample_id'] = test_id
        fold_csv['category_id'] = fold_result
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
    total_csv['sample_id'] = test_id
    total_csv['category_id'] = final_result
    total_csv.to_csv(os.path.join(output_dir, f'fusion_acc-{round(acc,4)}_f1-{round(f1,4)}.csv'),index=False)

if __name__ == '__main__':
    main()