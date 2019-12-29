import pandas as pd
import time
import glob
import torch
from models import *
import torch.nn as nn
import sklearn.metrics
import os
import numpy as np
import json
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import json
import random

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (9, 9),
         'axes.labelsize': 15,
         'axes.titlesize': 15}
mpl.rcParams.update(params)


def copy_file(src_path, dst_dir):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    src_file = src_path.split('/')[-1]
    dst_path = os.path.join(dst_dir, src_file)
    shutil.copyfile(src_path, dst_path)

def copy_dir(src, dst, symlinks=False, ignore=None):
    if not os.path.isdir(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def save_snapshot(model, optimizer, snapshot_dir, epoch, iteration, snapshot_epoch_fre):
    if epoch % snapshot_epoch_fre == 0:
        dir_name = 'snapshot/epoch-%04d-iter-%04d' % (epoch, iteration)
        save_dir = os.path.join(snapshot_dir, dir_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        optimizer_path = os.path.join(save_dir, 'optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
        # print('[SAVE] {} {}]' .format(model_path, optimizer_path))


def save_plot(outputs, targets, save_root, epoch, Flag='Test', title='', ID=None, initial=None, first_batch=False, ax=None):
    png_save_root = os.path.join(save_root + '/result/')
    if not os.path.isdir(png_save_root):
        os.makedirs(png_save_root)
    plt.close()

    fig, ax = plt.subplots()

    ax.set_title('{} {} {}epoch'.format(Flag, title, epoch))
    ax.set_xlabel('pred')
    ax.set_ylabel('target')

    ax.set_xlim(40, 250)
    ax.set_ylim(40, 250)

    ax.plot([40, 240], [50, 250], 'k-', alpha=0.3)
    ax.plot([40, 250], [40, 250], 'k-', alpha=0.3)
    ax.plot([50, 250], [40, 240], 'k-', alpha=0.3)

    ax.scatter(outputs.data.cpu().numpy(), targets.data.cpu().numpy(), s=2, alpha=1.0)

    return ax, plt


def pad_and_pack(data):
    padded = rnn_utils.pad_sequence([torch.tensor(x) for x in data]) #(seq_len, batch, feature)
    seq_lengths = [len(x) for x in data]
    packed = rnn_utils.pack_padded_sequence(padded, seq_lengths, enforce_sorted=False)
    return padded, packed

def pack(padded):
    seq_lengths = [len(x) for x in padded]
    print(seq_lengths)
    packed = rnn_utils.pack_padded_sequence(padded, seq_lengths, enforce_sorted=False)
    return packed

def create_toy_data(num_data, input_size, seed, seq_len=40):
    np.random.seed(seed)
    seq_len_list = [seq_len]
    X = np.random.uniform(-3,3,size=(1,seq_len,input_size))
    for i in range(num_data-1):
        random_seq_len = np.random.randint(1,seq_len)
        seq_len_list.append(random_seq_len)
        new_data = np.random.uniform(-1,1,size=(1,random_seq_len,input_size))
        new_data = np.concatenate((new_data, np.zeros((1,seq_len-random_seq_len,input_size))), axis=1)
        X = np.concatenate((X,new_data))
    y = np.array([[0 if x < 0 else 1 for x in np.cumsum(row)] for row in np.sum(X, axis=2)])
    print(y.sum() / y.size)
    # Shape (batch_size, seq_len, num_feature =1)
    return X, y, seq_len_list

def compute_f1score(true, pred, indent=False):
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    f1 = sklearn.metrics.f1_score(true, pred)
    precision = sklearn.metrics.precision_score(true,pred)
    recall = sklearn.metrics.recall_score(true,pred)
    if indent:
        print("    F1: {:.3f} , Precision: {:.3f}, Recall : {:.3f} \n".format(f1, precision, recall))
    else:
        print ("F1: {:.3f} , Precision: {:.3f}, Recall : {:.3f}".format(f1, precision, recall))

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval_regression(model, loader, device, log_dir, save_result=False, criterion=nn.L1Loss(reduction='none')):
    if save_result:
        f = open("{}/test_result.csv".format(log_dir), 'ab')
    with torch.no_grad():
        model.eval()
        sbp_running_loss = 0
        dbp_running_loss = 0
        total = 0
        for (inputs, targets) in loader:
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            targets = targets.float().view(-1, 2)
            if save_result:
                concat = torch.stack((outputs, targets), dim=1).squeeze(dim=-1)
                np.savetxt(f, concat.data.cpu().numpy())

            loss = criterion(outputs, targets)
            sbp_loss = loss[:,0]
            dbp_loss = loss[:,1]
            sbp_running_loss += sbp_loss.sum().item()
            dbp_running_loss += dbp_loss.sum().item()
            total += inputs.size(0)

        print("   L1 loss on Validation SBP: {:.4f}    DBP: {:.4f}".format(sbp_running_loss/total, sbp_running_loss/total))
    if save_result:
        f.close()
    return sbp_running_loss, dbp_running_loss, total

def eval_rnn_regression(model, loader, device, data_type, output_size, criterion, is_snapshot_epoch, save_result_root, epoch):
    with torch.no_grad():
        sbp_running_loss = 0
        dbp_running_loss = 0
        total = 0
        raw_sbp_loss = 0
        raw_dbp_loss = 0

        total_output = torch.tensor([]).to(device)
        total_target = torch.tensor([]).to(device)

        for i, (inputs, targets, seq_len) in enumerate(loader):
            inputs = inputs.permute(1,0,2).to(device)
            targets = targets.float().permute(1,0,2).to(device)
            seq_len = seq_len.to(device)

            outputs = model(inputs, seq_len, device)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                flattened_output = torch.cat([flattened_output, outputs[:seq,idx,:].view(-1,output_size)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq,idx,:].view(-1,output_size)), dim=0)

            if is_snapshot_epoch:
                total_output = torch.cat([total_output, flattened_output], dim=0)
                total_target = torch.cat([total_target, flattened_target], dim=0)

            sbp_loss = criterion(flattened_target[:, 0], flattened_output[:, 0])
            dbp_loss = criterion(flattened_target[:, 1], flattened_output[:, 1])

            sbp_running_loss += sbp_loss.item()
            dbp_running_loss += dbp_loss.item()

            total += len(seq_len)

        print("Validation, SBP_Loss: {:.4f} DBP_Loss: {:.4f}".format(sbp_running_loss / total, sbp_running_loss / total))

        if is_snapshot_epoch:
            total_sub = total_output - total_target
            raw_sbp_loss = torch.sum(abs(total_sub[:,0]))
            raw_dbp_loss = torch.sum(abs(total_sub[:,1]))

            sample_idx = np.random.choice(range(total), size=50, replace=False)
            sample_output, sample_target = un_normalize(total_output[sample_idx, :], total_target[sample_idx, :], 'RNN', device)

            ax, plt = save_plot(sample_output[:,0], sample_target[:,0], save_result_root, epoch + 1, data_type, 'SBP')
            plt.savefig(save_result_root + '/result/' + 'sbp_{}epoch_{}_{}loss.png'.format(epoch + 1, data_type, raw_sbp_loss / total), dpi=300)
            ax, plt = save_plot(sample_output[:,1], sample_target[:,1], save_result_root, epoch + 1, data_type, 'DBP')
            plt.savefig(save_result_root + '/result/' + 'dbp_{}epoch_{}_{}loss.png'.format(epoch + 1, data_type, raw_dbp_loss / total), dpi=300)

    return sbp_running_loss, dbp_running_loss, total, raw_sbp_loss, raw_dbp_loss




def un_normalize(outputs, targets, model_type, device):
    with open('./tensor_data/{}/mean_value.json'.format(model_type)) as f:
        mean = json.load(f)
        mean = torch.tensor([mean['VS_sbp'], mean['VS_dbp']]).to(device)

    with open('./tensor_data/{}/std_value.json'.format(model_type)) as f:
        std = json.load(f)
        std = torch.tensor([std['VS_sbp'], std['VS_dbp']]).to(device)

    outputs = torch.add(torch.mul(outputs, std), mean)
    targets = torch.add(torch.mul(targets, std), mean)

    return outputs, targets


def confusion_matrix(preds, labels, n_classes):

    conf_matrix = torch.zeros(n_classes, n_classes)
    for p, t in zip(preds, labels):
        conf_matrix[int(p.item()), int(t.item())] += 1

    sensitivity_log = {}
    specificity_log = {}
    TP = conf_matrix.diag()

    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c]+FN))
        specificity = (TN / (TN+FP))
        sensitivity_log['class_{}'.format(c)] = sensitivity
        specificity_log['class_{}'.format(c)] = specificity

        #print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))
        #print('Sensitivity = {:.4f}'.format(sensitivity))
        #print('Specificity = {:.4f}'.format(specificity))
        #print('\n')

    return conf_matrix, (sensitivity_log, specificity_log)




def un_normalize(outputs, targets, model_type, device):
    with open('./tensor_data/{}/mean_value.json'.format(model_type)) as f:
        mean = json.load(f)
        mean = torch.tensor([mean['VS_sbp'], mean['VS_dbp']]).to(device)

    with open('./tensor_data/{}/std_value.json'.format(model_type)) as f:
        std = json.load(f)
        std = torch.tensor([std['VS_sbp'], std['VS_dbp']]).to(device)

    outputs = torch.add(torch.mul(outputs, std), mean)
    targets = torch.add(torch.mul(targets, std), mean)

    return outputs, targets


def rnn_load_data(path, target_idx):
    data = torch.load(path)
    seq_len_list = torch.LongTensor([len(x) for x in data])
    seq_len_list, perm_idx = seq_len_list.sort(0, descending=True)
    padded = rnn_utils.pad_sequence([torch.tensor(x) for x in data])
    padded = padded[:, perm_idx, :]
    X = padded[:, :, :-4]
    y = padded[:, :, target_idx]

    return X, y, seq_len_list

def load_checkpoint(state, model, optimizer):
    model = model
    optimizer = optimizer
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    return model, optimizer

def make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def load_json_data(PATH):
    """
    약 3분 소요
    Inputs:
        - PATH: relative path to folder containing data (.json)
    Outputs :
        - Dataframe with 188 features and 3 targets
    """

    print("Loading Data...")
    start_time = time.time()
    df = None

    for i in glob.glob(PATH+'/*.json'):
        df = pd.concat([df, pd.read_json(i)])

    df.reset_index(drop=True, inplace=True)

    print('Time Elapsed to Load Data: ', time.time()- start_time)

    return df

def save_as_tensor(PATH):
    """
    Saves 12 tensors as .pt for train,val,test data and three types of target data
        - X_Training, X_Validation, X_Test
        - y_Training_clas_target, [...], [...]
        - [...], y_Validation_VS_sbp_target, [...]
        - [...], [...], y_Test_VS_dbp_target

    Inputs:
        - PATH : relative path to folder containing data (.json)
    """

    df = load_json_data(PATH)

    print('Converting to Tensor...')
    start_time = time.time()
    for type in ['Training','Test','Validation']:
        data = df.loc[df.ID_class == type]

        for t in targets_list: # 3 가지 target_type 저장
            y = torch.tensor(data[t].values, dtype=torch.float)
            torch.save(y, 'data/123y_%s_%s.pt' % (type, t))

        X = torch.tensor(data.drop(labels=id_features+targets_list, axis=1).values, dtype=torch.float)
        torch.save(X, 'data/123X_%s.pt' %(type))
        print("    Finished converting {} Data , Duration {}".format(type, time.time()-start_time))



def data_preproc_to_remove_normal_after_abnormal(dataset):
    dataset = np.asarray(dataset)
    new_dataset = list()
    for idx in range(len(dataset)):
        cls_target_ = dataset[idx][:,-2:]
        # sbp ########################################################
        sbp_sort = np.sort(cls_target_[:,0])
        sbp_sort_idx = np.argsort(cls_target_[:,0])
        sbp_count_under_6 = (np.sort(cls_target_[:,0]) < 6).sum()
        if sbp_count_under_6 != len(cls_target_) : sbp_idx_abnormal_6 = sbp_sort_idx[sbp_count_under_6]
        else : sbp_idx_abnormal_6 = 100000000000
        if sbp_sort[0] == 0 : sbp_idx_abnormal_0 = sbp_sort_idx[0]
        else : sbp_idx_abnormal_0 = 100000000000

        if sbp_idx_abnormal_0 > sbp_idx_abnormal_6 : sbp_idx_abnormal = sbp_idx_abnormal_6
        else : sbp_idx_abnormal = sbp_idx_abnormal_0
        #############################################################

        # dbp
        dbp_sort = np.sort(cls_target_[:,1])
        dbp_sort_idx = np.argsort(cls_target_[:,1])
        dbp_count_under_4 = (np.sort(cls_target_[:,1]) < 4).sum()
        if dbp_count_under_4 != len(cls_target_) : dbp_idx_abnormal_4 = dbp_sort_idx[dbp_count_under_4]
        else : dbp_idx_abnormal_4 = 100000000000
        if dbp_sort[0] == 0 : dbp_idx_abnormal_0 = dbp_sort_idx[0]
        else : dbp_idx_abnormal_0 = 100000000000

        if dbp_idx_abnormal_0 > dbp_idx_abnormal_4 : dbp_idx_abnormal = dbp_idx_abnormal_4
        else : dbp_idx_abnormal = dbp_idx_abnormal_0

        if sbp_idx_abnormal < dbp_idx_abnormal : idx_abnormal = sbp_idx_abnormal
        else : idx_abnormal = dbp_idx_abnormal



        if idx_abnormal != 100000000000 :
            new_dataset.append(dataset[idx][:idx_abnormal+1])
        else:
            new_dataset.append(dataset[idx])

    new_dataset = np.asarray(new_dataset)
    torch.save(new_dataset, './tensor_data/RNN/train_preproc.pt')
    
    return dataset

##################################################################################
##################################################################################
##################################################################################
'''
여기부터가 내가 작성하거나 수정한 code인듯
'''

# 이 eval 부분은 v2까지 사용했던 것
def eval_rnn_classification(loader, model, device, output_size, criterion1, criterion2, num_class1, num_class2, log_dir=None, epoch=None):
    with torch.no_grad():
        running_loss = 0
        total = 0
        val_correct1 = 0
        val_correct2 = 0
        val_total = 0

        total_output1 = torch.tensor([], dtype=torch.long).to(device)
        total_output2 = torch.tensor([], dtype=torch.long).to(device)
        total_target = torch.tensor([]).to(device)

        # accum_test_dict = dict()    # 이 주석들은 def test_one_hour 를 위한 부분
        # accum_test_dict['tp'] = 0   # 가장 최초의 classification으로 학습 후, 1시간 안에 문제가 발생하는지를 체크하려고 만든 것
        # accum_test_dict['fp'] = 0
        # accum_test_dict['tn'] = 0
        # accum_test_dict['fn'] = 0

        for i, (inputs, targets, seq_len) in enumerate(loader):
            inputs = inputs.permute(1, 0, 2).to(device)
            targets = targets.float().permute(1, 0, 2).to(device)
            seq_len = seq_len.to(device)

            output1, output2 = model(inputs, seq_len, device)

            # test_dict = test_one_hour(inputs, output1, output2, targets, seq_len, 'RNN')  # 이 주석들은 def test_one_hour 를 위한 부분
            # accum_test_dict['tp'] += test_dict['tp']
            # accum_test_dict['fp'] += test_dict['fp']
            # accum_test_dict['tn'] += test_dict['tn']
            # accum_test_dict['fn'] += test_dict['fn']


            flattened_output1 = torch.tensor([]).to(device)
            flattened_output2 = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                flattened_output1 = torch.cat([flattened_output1, output1[:seq, idx, :].reshape(-1, num_class1)], dim=0)
                flattened_output2 = torch.cat([flattened_output2, output2[:seq, idx, :].reshape(-1, num_class2)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, output_size)), dim=0)

            loss1 = criterion1(flattened_output1, flattened_target[:, 0].long())
            loss2 = criterion2(flattened_output2, flattened_target[:, 1].long())
            loss = loss1 + loss2
            # total += len(seq_len)
            running_loss += loss.item()

            _, pred1 = torch.max(flattened_output1, 1)
            _, pred2 = torch.max(flattened_output2, 1)
            val_correct1 += (pred1 == flattened_target[:, 0].long()).sum().item()
            val_correct2 += (pred2 == flattened_target[:, 1].long()).sum().item()
            val_total += len(pred1)

            total_output1 = torch.cat([total_output1, pred1], dim=0)
            total_output2 = torch.cat([total_output2, pred2], dim=0)
            total_target = torch.cat([total_target, flattened_target], dim=0)

            if i < 100:
                save_result_txt(torch.argmax(output1.permute(1,0,2), dim=2), targets[:,:, 0].permute(1,0), log_dir+'/txt/', epoch, 'val_sbp', seq_lens=seq_len)
                save_result_txt(torch.argmax(output2.permute(1,0,2), dim=2), targets[:,:, 1].permute(1,0), log_dir+'/txt/', epoch, 'val_dbp', seq_lens=seq_len)
        # print(accum_test_dict)

        print("\tEvaluated Loss : {:.4f}".format(running_loss / i), end=' ')
        print("\tAccuracy of Sbp : {:.2f}% Dbp : {:.2f}%".format(100 * val_correct1 / val_total, 100 * val_correct2 / val_total))

    return running_loss/i, i, total_output1, total_output2, total_target, 100 * val_correct1 / val_total, 100 * val_correct2 / val_total

# v3 : 세 기준에 의한 binary classification
def confusion_matrix_save_as_img(matrix, save_dir, epoch=0, iteration=0, data_type='train', name=None, v3=False):
    mpl.use('Agg')

    save_dir = save_dir + '/confusion_matrix'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not v3:
        if name == 'sbp':
            num_class = 7
        else :
            num_class = 5
    else : # binary 문제로 바꾼 부분 option 
        num_class = 2
    matrix = np.transpose(matrix)


    df_cm = pd.DataFrame(matrix.astype(int), index = [str(i) for i in range(num_class)], columns = [str(i) for i in range(num_class)])
    plt.figure(figsize = (9,7))
    ax = sn.heatmap(df_cm, annot=True, cmap='RdBu_r', vmin=0, vmax=10000, fmt="d", annot_kws={"size": 15})
    ax.set_title('{}_{}epoch_count'.format(name, epoch))
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label')
    plt.xlabel('Pred. label')

    ax.figure.savefig('{}/{}_{}_{}Epoch_Count.jpg'.format(save_dir, data_type, name, epoch))
    # ax.figure.savefig('{}/{}_{}_{}epoch_{}iter_count.jpg'.format(save_dir, data_type, name, epoch, iteration))
    plt.close("all")


    matrix_sum = matrix.sum(axis=1)
    for i in range(len(matrix)):
        matrix[i] /= matrix_sum[i]

    df_cm = pd.DataFrame(matrix, index = [str(i) for i in range(num_class)], columns = [str(i) for i in range(num_class)])
    plt.figure(figsize = (9,7))
    ax = sn.heatmap(df_cm, annot=True, cmap='RdBu_r', vmin=0, vmax=1, annot_kws={"size": 15})
    ax.set_title('{}_{}Epoch'.format(name, epoch))
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label')
    plt.xlabel('Pred. label')

    ax.figure.savefig('{}/{}_{}_{}Epoch.jpg'.format(save_dir, data_type, name, epoch))
    # ax.figure.savefig('{}/{}_{}_{}epoch_{}iter.jpg'.format(save_dir, data_type, name, epoch, iteration))
    plt.close("all")


# many-to-one 학습의 feasibility를 보기 위해
# data 가공하기 위한 code
# frame 개수를 5개씩 끊어서 저장
# seq 길이가 더 길어져서 용량이 커짐
def data_preproc_to_cut_specific_len(dataset, cut_len=5, d_type='Train'):
    dataset = np.asarray(dataset)
    
    new_dataset = list()
    for idx in range(len(dataset)):
        if len(dataset[idx])<=5:
            new_dataset.append(dataset[idx])
        else:
            length = len(dataset[idx]) - cut_len + 1
            for i in range(length):
                temp = dataset[idx][i:i+cut_len]
                new_dataset.append(temp)


    new_dataset = np.asarray(new_dataset)
    torch.save(new_dataset, './tensor_data/RNN/{}_{}cut.pt'.format(d_type, cut_len))
    
    return dataset



# e.g. target==0 인 상황
# target 0 으로 맞추거나, 최소한 target 1로 맞출길 바라서,
# target class를 변경해줌
# 기존 (0) ->  변경 후 (0.8, 0.2, 0, 0, 0, 0, 0)
def smooth_one_hot(target, smoothing=0.1, num_class=7):
    smooth_target = torch.zeros((target.size(0), num_class))
    zero_target = (target == 0)
    last_target = (target == num_class-1)
    perfect_normal_target = (target == int((num_class-1)/2) )
    normal_target = (target != 0) & (target != num_class-1) & (target != int((num_class-1)/2) ) 


    smooth_target[zero_target,0] = 1 - 2*smoothing
    smooth_target[zero_target,1] = 2*smoothing

    smooth_target[last_target, num_class-1] = 1 - 2*smoothing
    smooth_target[last_target, num_class-2] = 2*smoothing

    smooth_target[perfect_normal_target, int((num_class-1)/2)] = 1 - 4*smoothing
    smooth_target[perfect_normal_target, int((num_class-1)/2) +1] = 2*smoothing
    smooth_target[perfect_normal_target, int((num_class-1)/2) -1] = 2*smoothing

    smooth_target[normal_target, target[normal_target].long()] = 1-3*smoothing
    smooth_target[normal_target, (target[normal_target]-1).long()] = 1.5*smoothing
    smooth_target[normal_target, (target[normal_target]+1).long()] = 1.5*smoothing

    return smooth_target.float()


# 정성적으로 눈으로 보기 위해 txt로 저장
def save_result_txt(outputs, targets, save_root, epoch, Flag='Test', ID=None, initial=None, seq_lens=None):
    txt_save_root = os.path.join(save_root)
    if not os.path.isdir(txt_save_root):
        os.makedirs(txt_save_root)
    with open(txt_save_root+'result_{}epoch_{}.txt'.format(epoch, Flag), 'a') as f :
        if ID is not None:
            f.write('ID : {}\n'.format(str(ID.data.cpu().numpy())))
        f.write('{:40}    {:40}\n'.format('target', 'pred'))
        if initial is not None :
            f.write('{:40}  <<-- initial [sbp,dbp]\n'.format(str(initial.int().data.cpu().numpy())))
        if seq_lens is None:
            for output, target in zip(outputs, targets):
                f.write('{:40}  {:40}'.format(  str(target.int().data.cpu().numpy()) ,str(output.int().data.cpu().numpy())   ))
                f.write('\n')
        else :
            for output, target,seq_len in zip(outputs, targets, seq_lens):
                f.write('{:40}  {:40}  {:5}'.format(  str(target[:seq_len].int().data.cpu().numpy()) ,str(output[:seq_len].int().data.cpu().numpy()), str(seq_len.item())   ))
                f.write('\n')
        f.write('\n')


def test_one_hour(batch_inputs, batch_outputs_sbp, batch_outputs_map, batch_targets, seq_len=None, model_type='RNN', crite=60) :
    """
    Arguments:
    batch_inputs, batch_outputs_sbp, batch_outputs_map, batch_targets : torch.tensor
    seq_len : list
    model_type : str
    
    batch_inputs shape : (padded_seqence_length, batch size, feature_size)
    batch_outputs_sbp : model's output
    batch_outputs_sbp shape : (padded_seqence_length, batch_size, sbp_num_classes)
    batch_outputs_map shape : (padded_seqence_length, batch_size, map_num_classes)
    batch_targets shape : (padded_seqence_length, batch size, 2)

    seq_len shape : [int, int, int, int, ....]
    len(seq_len) -> mini-batch size

    Outputs:
    dictionary : {'tp':int , 'fp':int, 'tn':int, 'fn':int} 

    e.g.)
    return_dict = {'tp':4 , 'fp':5, 'tn':3, 'fn':7}
    
    print(output_dict['tp'])
    4

    cf)
    mean : "HD_ntime": 31.632980458822722, "HD_ctime": 112.71313384332716
    std : "HD_ntime": 24.563710661941634, "HD_ctime": 78.88638128125473
    """

    # mean_HD_ntime = 31.632980458822722
    # std_HD_ntime = 24.563710661941634
    mean_HD_ctime = 112.71313384332716
    std_HD_ctime = 78.88638128125473
    
    tp, fp, tn, fn = 0,0,0,0

    return_dict = dict()
    return_dict['tp'] = 0
    return_dict['fp'] = 0
    return_dict['tn'] = 0
    return_dict['fn'] = 0

    # error check
    if model_type=='RNN' and seq_len is None:
        print('test error. model type:RNN / but seq_len_list is absent.')
        assert()
    if seq_len is not None :
        # TODO: absolute sbp under 90 ----> diff between init & current??
        _, preds_sbp = torch.max(batch_outputs_sbp, 2)  # pred shape : (seq_len, batch size)
        _, preds_map = torch.max(batch_outputs_map, 2)  # pred shape : (seq_len, batch size)
        
        c_time_tensor = (batch_inputs[:,:,5])   # shape : (seq_len, batch size)
        padd_flag = (c_time_tensor == 0)
        c_time_tensor[padd_flag] = 999          # 999 is big enough to seperate real & pad / because : c_time_tensor -> normalized value
        c_time_tensor = (c_time_tensor * std_HD_ctime + mean_HD_ctime).int()    # un-normalize

        # n_time_tensor = (batch_inputs.permute(1,0,2)[:,:,4])
        # n_time_tensor[padd_flag] = 999
        # n_time_tensor = (n_time_tensor * std_HD_ntime + mean_HD_ntime).int()


        max_seq = max(seq_len)
        for idx in range(max_seq):
            criterion_flag = c_time_tensor[idx].detach().expand_as(c_time_tensor) +60 >= c_time_tensor
            criterion_flag_count = criterion_flag.int().sum(0)  # shape : torch.Size([batch_size])

            gt_sbp_exist_list = [1 if torch.sum((batch_targets[idx:idx+criterion_flag_count[i],i,0]==0).int()) else 0 for i in range(len(criterion_flag_count)) ]
            gt_map_exist_list = [1 if torch.sum((batch_targets[idx:idx+criterion_flag_count[i],i,1]==0).int()) else 0 for i in range(len(criterion_flag_count)) ]
            gt_exist_list = [gt_sbp_exist_list[i] or gt_map_exist_list[i] for i in range(len(gt_sbp_exist_list))]
            
            pred_sbp_exist_list = [1 if torch.sum((preds_sbp[idx:idx+criterion_flag_count[i],i]==0).int()) else 0 for i in range(len(criterion_flag_count)) ]
            pred_map_exist_list = [1 if torch.sum((preds_map[idx:idx+criterion_flag_count[i],i]==0).int()) else 0 for i in range(len(criterion_flag_count)) ]
            pred_exist_list = [pred_sbp_exist_list[i] or pred_map_exist_list[i] for i in range(len(pred_sbp_exist_list))]
            
            tn, fp, fn, tp = (sklearn.metrics.confusion_matrix(gt_exist_list, pred_exist_list, labels=[0,1])).ravel()
            return_dict['tp'] += tp
            return_dict['fp'] += fp
            return_dict['fn'] += fn
            return_dict['tn'] += tn
        return return_dict


# 60분 단위로 data를 바꾸기 위한 부분
# TODO : interpolation 부분이 아닌, real만 반영한 target
# TODO : 현재값을 제외하고, 앞으로의 값만 반영한 target
def data_modify_same_ntime(data, ntime=60, d_type='Train', base_dir=None, mask=True):
   new_data = np.copy(data)
   if base_dir is None:
       mean_HD_ctime = 112.71313384332716
       std_HD_ctime = 78.88638128125473
       mean_VS_sbp = 132.28494659660691
       mean_VS_dbp = 72.38785072807198
       std_VS_sbp = 26.863242507719363
       std_VS_dbp = 14.179094454260184
       idx_HD_ctime = 7
       idx_VS_sbp = 12
       idx_VS_dbp = 13

   else:
       with open(os.path.join(base_dir, 'mean_value.json')) as mean, open(os.path.join(base_dir, 'std_value.json')) as std, open(os.path.join(base_dir, 'columns.csv')) as columns:
           mean_data = json.load(mean)
           std_data = json.load(std)
           mean_HD_ctime = mean_data['HD_ctime']
           mean_VS_sbp = mean_data['VS_sbp']
           mean_VS_dbp = mean_data['VS_dbp']
           std_HD_ctime = std_data['HD_ctime']
           std_VS_sbp = std_data['VS_sbp']
           std_VS_dbp = std_data['VS_dbp']
           for idx, col in enumerate(columns.readlines()):
               col = col.strip()
               if col == 'VS_sbp':
                   idx_VS_sbp = idx
               elif col == 'VS_dbp':
                   idx_VS_dbp = idx
               elif col == 'HD_ctime':
                   idx_HD_ctime = idx

   c_time_list = [(data[i][:,idx_HD_ctime]*std_HD_ctime+mean_HD_ctime).astype(int) for i in range(len(data))]
   sbp_list =[(data[i][:,idx_VS_sbp]*std_VS_sbp+mean_VS_sbp).astype(int) for i in range(len(data))]
   map_list =[((data[i][:,idx_VS_dbp]*std_VS_dbp+mean_VS_dbp) / 3. + (data[i][:,idx_VS_sbp]*std_VS_sbp+mean_VS_sbp)* 2. / 3.).astype(int) for i in range(len(data))]
   # 각 frame의 c_time, sbp, map를 받았음.
   # <<<<중요>>>>> 7,12,13은 data 형태에 따라서 바꿔줘야 함

   for data_idx in range(len(data)):
       if data_idx == 1000:
           print('{}_{}'.format(d_type, data_idx))
       sbp_absolute_value = sbp_list[data_idx]
       map_absolute_value = map_list[data_idx]
       sbp_init_value = sbp_list[data_idx][0]
       map_init_value = map_list[data_idx][0]
       dbp_init_value = data[data_idx][0,idx_VS_dbp] * std_VS_dbp + mean_VS_dbp
       sbp_diff = sbp_absolute_value - sbp_init_value
       map_diff = map_list[data_idx] - map_init_value

       if mask:
           real_flag = [x[0] == 1 for x in data[data_idx]]
           temp_data_concat = np.zeros((len(sbp_diff), 10))
       else:
           real_flag = [1 for _ in data[data_idx]]
           temp_data_concat = np.zeros((len(sbp_diff), 5))

       # print()
       # print('c_time_list[{}]: '.format(data_idx), c_time_list[data_idx])

       # print('sbp_list[{}]:'.format(data_idx), sbp_list[data_idx])
       # print('map_list[{}]:'.format(data_idx), map_list[data_idx])
       # print('sbp_init_value[{}]:'.format(data_idx), sbp_init_value)
       # print('map_init_value[{}]:'.format(data_idx), map_init_value)
       # print('sbp_diff[{}]:'.format(data_idx), sbp_diff)
       # print('map_diff[{}]:'.format(data_idx), map_diff)

       for frame_idx in range(len(data[data_idx])):
           criterion_flag = (c_time_list[data_idx][frame_idx] + ntime >= c_time_list[data_idx]) & (c_time_list[data_idx][frame_idx] < c_time_list[data_idx]) # shape : [True, True, True, True, False, False, False, ....]

           if np.sum(((sbp_absolute_value[criterion_flag] - sbp_absolute_value[frame_idx]) <= -20).astype(int)) : curr_sbp_exist = 1
           else : curr_sbp_exist = 0
           if np.sum(((map_absolute_value[criterion_flag] - map_absolute_value[frame_idx]) <= -10).astype(int)) : curr_map_exist = 1
           else : curr_map_exist = 0

           if np.sum(((sbp_absolute_value[criterion_flag & real_flag] - sbp_absolute_value[frame_idx]) <= -20).astype(int)) : curr_real_sbp_exist = 1
           else : curr_real_sbp_exist = 0
           if np.sum(((map_absolute_value[criterion_flag & real_flag] - map_absolute_value[frame_idx]) <= -10).astype(int)) : curr_real_map_exist = 1
           else : curr_real_map_exist = 0

           if np.sum((sbp_diff[criterion_flag]<=-20).astype(int)) : sbp_exist = 1    # 초기값 대비 20 이상 떨어지는게 존재하면 1, else 0.
           else : sbp_exist = 0
           if np.sum((map_diff[criterion_flag]<=-10).astype(int)) : map_exist = 1
           else : map_exist = 0
           if np.sum((sbp_absolute_value[criterion_flag]<90).astype(int)) : sbp_under_90 = 1
           else : sbp_under_90 = 0

           if np.sum((sbp_diff[criterion_flag & real_flag]<=-20).astype(int)) : real_sbp_exist = 1    # 초기값 대비 20 이상 떨어지는게 존재하면 1, else 0.
           else : real_sbp_exist = 0
           if np.sum((map_diff[criterion_flag & real_flag]<=-10).astype(int)) : real_map_exist = 1
           else : real_map_exist = 0
           if np.sum((sbp_absolute_value[criterion_flag & real_flag]<90).astype(int)) : real_sbp_under_90 = 1
           else : real_sbp_under_90 = 0

           if frame_idx == (len(data[data_idx])-1):    # sbp_list가 각 frame의 sbp를 가져왔는데, 마지막 frame은 다음 target frame을 반영해줘야 했었음.
               last_target_sbp = (data[data_idx][-1,-4]*std_VS_sbp + sbp_init_value ).astype(int)
               last_target_map = ((data[data_idx][-1,-3]*std_VS_dbp + dbp_init_value) / 3. + (data[data_idx][-1,-4]*std_VS_sbp + sbp_init_value) * 2. / 3.).astype(int)
               if last_target_sbp - sbp_init_value <= -20 : sbp_exist = 1
               else: sbp_exist = 0
               if last_target_map - map_init_value <= -10 : map_exist = 1
               else: map_exist = 0
               if last_target_sbp < -90 : sbp_under_90 = 1
               else: sbp_under_90 = 0

           if mask:
               temp_data_concat[frame_idx] = np.array((sbp_exist, map_exist, sbp_under_90, real_sbp_exist, real_map_exist, real_sbp_under_90, curr_sbp_exist, curr_map_exist, curr_real_sbp_exist, curr_real_map_exist))
           else:
               temp_data_concat[frame_idx] = np.array((sbp_exist, map_exist, sbp_under_90, curr_sbp_exist, curr_map_exist))

       new_data[data_idx] = np.concatenate((new_data[data_idx], temp_data_concat), axis=1)

   # torch.save(data, 'data/tensor_data/1227_EF_60min/{}_{}min.pt'.format(d_type, ntime)) # save root 잘 지정해줄 것
   return new_data


# version3 용 eval.
def eval_rnn_classification_v3(loader, model, device, output_size, criterion, threshold=0.5, log_dir=None, epoch=None, step=0):
    with torch.no_grad():
        running_loss = 0
        total_output = torch.tensor([], dtype=torch.float).to(device)
        total_target = torch.tensor([]).to(device)

        for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(loader):
            inputs_fix = inputs_fix.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = targets_real.float().to(device)
            output = model(inputs_fix, inputs_seq, seq_len, device) # shape : (seq, batch size, 5)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                flattened_output = torch.cat([flattened_output, output[:seq, idx, :].reshape(-1, output_size)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, output_size)), dim=0)

            loss_sbp = criterion(flattened_output[:,0], flattened_target[:,0])
            loss_map = criterion(flattened_output[:,1], flattened_target[:,1])
            loss_under90 = criterion(flattened_output[:,2], flattened_target[:,2])
            loss_sbp2 = criterion(output[:,3], targets[:,3])
            loss_map2 = criterion(output[:,4], targets[:,4])

            loss = loss_sbp + loss_map + loss_under90 + loss_sbp2 + loss_map2
            running_loss += loss.item()

            total_target = torch.cat([total_target, flattened_target], dim=0)
            total_output = torch.cat([total_output, flattened_output], dim=0)
        print("\tEvaluated Loss : {:.4f}".format(running_loss / i))
        confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', log_dir, epoch, step, composite=False, cal_roc=True)

        val_total = len(total_output)

        for thres in threshold:
            val_correct0, val_correct1, val_correct2, val_correct3, val_correct4 = 0, 0, 0, 0, 0

            pred0 = (F.sigmoid(total_output[:,0]) > thres).long()
            pred1 = (F.sigmoid(total_output[:,1]) > thres).long()
            pred2 = (F.sigmoid(total_output[:,2]) > thres).long()
            pred3 = (F.sigmoid(total_output[:,3]) > thres).long()
            pred4 = (F.sigmoid(total_output[:,4]) > thres).long()

            val_correct0 += (pred0 == total_target[:,0].long()).sum().item()
            val_correct1 += (pred1 == total_target[:,1].long()).sum().item()
            val_correct2 += (pred2 == total_target[:,2].long()).sum().item()
            val_correct3 += (pred3 == total_target[:,3].long()).sum().item()
            val_correct4 += (pred4 == total_target[:,4].long()).sum().item()

            sbp_confusion_matrix, sbp_log = confusion_matrix(pred0, total_target[:,0], 2)
            map_confusion_matrix, dbp_log = confusion_matrix(pred1, total_target[:, 1], 2)
            under90_confusion_matrix, dbp_log = confusion_matrix(pred2, total_target[:, 2], 2)
            sbp2_confusion_matrix, dbp_log = confusion_matrix(pred3, total_target[:, 3], 2)
            map2_confusion_matrix, dbp_log = confusion_matrix(pred4, total_target[:, 4], 2)
            confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres),
                                               epoch, step, 'val', 'sbp', v3=True)
            confusion_matrix_save_as_img(map_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres),
                                               epoch, step, 'val', 'map', v3=True)
            confusion_matrix_save_as_img(under90_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres), epoch, step, 'val', 'under90', v3=True)
            confusion_matrix_save_as_img(sbp2_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres), epoch, step, 'val', 'sbp2', v3=True)
            confusion_matrix_save_as_img(map2_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres), epoch, step, 'val', 'map2', v3=True)


            print("\t Threshold: {} \tAccuracy of SBP: {:.2f}%\t MAP: {:.2f}%\t Under90: {:.2f}% \t SBP2: {:.2f}% \t MAP2: {:.2f}%".format(thres, 100 * val_correct0 / val_total,
                                                                                        100 * val_correct1 / val_total,
                                                                                        100 * val_correct2 / val_total,
                                                                                        100 * val_correct3 / val_total,
                                                                                        100 * val_correct4 / val_total))
            # print("\t Threshold: {} \tAccuracy of SBP: {:.2f}%\t MAP: {:.2f}%\t Under90: {:.2f}% \t".format(thres, 100 * val_correct0 / val_total, 100 * val_correct1 / val_total, 100 * val_correct2 / val_total))
def eval_rnn_classification_composite(loader, model, device, output_size, criterion, threshold=0.5, log_dir=None, epoch=None, step=0):
    with torch.no_grad():
        running_loss = 0
        total_output = torch.tensor([], dtype=torch.float).to(device)
        total_target = torch.tensor([]).to(device)

        for i, ((inputs_fix, inputs_seq), (targets, composite_targets), seq_len) in enumerate(loader):
            inputs_fix = inputs_fix.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = composite_targets.float().to(device)
            output = model(inputs_fix, inputs_seq, seq_len, device) # shape : (seq, batch size, 5)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                flattened_output = torch.cat([flattened_output, output[:seq, idx, :].reshape(-1, output_size)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, output_size)), dim=0)

            loss_init = criterion(flattened_output[:, 0], flattened_target[:,0])  # 이 loss는 알아서 input에 sigmoid를 씌워줌. 그래서 input : """logit""" / 단, target : 0 or 1
            loss_curr = criterion(flattened_output[:, 1], flattened_target[:,1])
            loss_under90 = criterion(flattened_output[:, 2], flattened_target[:,2])
            loss = loss_init + loss_curr + loss_under90

            running_loss += loss.item()

            total_target = torch.cat([total_target, flattened_target], dim=0)
            total_output = torch.cat([total_output, flattened_output], dim=0)
        print("\tEvaluated Loss : {:.4f}".format(running_loss / i))
        confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', log_dir, epoch, step, composite=True, cal_roc=True)

        val_total = len(total_output)

        for thres in threshold:
            val_correct0, val_correct1, val_correct2 = 0, 0, 0

            pred0 = (F.sigmoid(total_output[:,0]) > thres).long()
            pred1 = (F.sigmoid(total_output[:,1]) > thres).long()
            pred2 = (F.sigmoid(total_output[:,2]) > thres).long()

            val_correct0 += (pred0 == total_target[:,0].long()).sum().item()
            val_correct1 += (pred1 == total_target[:,1].long()).sum().item()
            val_correct2 += (pred2 == total_target[:,2].long()).sum().item()

            init_confusion_matrix, sbp_log = confusion_matrix(pred0, total_target[:,0], 2)
            curr_confusion_matrix, dbp_log = confusion_matrix(pred1, total_target[:, 1], 2)
            under90_confusion_matrix, dbp_log = confusion_matrix(pred2, total_target[:, 2], 2)

            confusion_matrix_save_as_img(init_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres),
                                               epoch, step, 'val', 'Initial', v3=True)
            confusion_matrix_save_as_img(curr_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres),
                                               epoch, step, 'val', 'Current', v3=True)
            confusion_matrix_save_as_img(under90_confusion_matrix.detach().cpu().numpy(),
                                               log_dir + '/{}'.format(thres), epoch, step, 'val', 'Under90', v3=True)


            print("\t Threshold: {} \tAccuracy of Initial: {:.2f}%\t Current: {:.2f}%\t Under90: {:.2f}% ".format(thres, 100 * val_correct0 / val_total,
                                                                                        100 * val_correct1 / val_total,
                                                                                        100 * val_correct2 / val_total))
            # print("\t Threshold: {} \tAccuracy of SBP: {:.2f}%\t MAP: {:.2f}%\t Under90: {:.2f}% \t".format(thres, 100 * val_correct0 / val_total, 100 * val_correct1 / val_total, 100 * val_correct2 / val_total))


def confidence_save_and_cal_auroc(mini_batch_outputs, mini_batch_targets, data_type, save_dir, epoch=9999, step=0, composite=True, cal_roc=True):
    '''
    mini_batch_outputs shape : (data 개수, 3) -> flattend_output    / cf. data개수 --> 각 투석의 seq_len의 total sum
    data_type : Train / Validation / Test
    minibatch 별로 저장 될 수 있게 open(,'a')로 했는데, 저장 다 하면 f.close() 권하긴 함.
    KY: Batch 별로 작동되게 수정하여 f.close() 추가
    '''
    print("Making roc curve...")
    save_dir += '/auroc'
    if composite:
        category = {'Initial':0, 'Current':1, 'Under90':2}
    else:
        category = {'sbp':0, 'map':1, 'under90':2, 'sbp2':3, 'map2':4}

    for key, value in category.items():
        key_dir = save_dir + '/' + key
        if not os.path.isdir(key_dir):
            os.makedirs(key_dir)
        f = open('{}/confidence_{}_{}_{}_{}.txt'.format(key_dir, epoch, step, data_type, key), 'w')
        for i in range(len(mini_batch_outputs[:,value])):
            f.write("{}\t{}\n".format(mini_batch_outputs[i,value].item(), mini_batch_targets[i,value].item()))
        f.close()
        if cal_roc :
            auroc = roc_curve_plot(key_dir, key, data_type, epoch, step)


def roc_curve_plot(load_dir, category='sbp', data_type='Validation', epoch=None, step= 0, save_dir=None):
    # calculate the AUROC
    # f1 = open('{}/Update_tpr.txt'.format(load_dir), 'w')
    # f2 = open('{}/Update_fpr.txt'.format(load_dir), 'w')

    conf_and_target_array = np.loadtxt('{}/confidence_{}_{}_{}_{}.txt'.format(load_dir, epoch, step, data_type, category),
                                       delimiter=',', dtype=np.str)
    file_ = np.array(
        [np.array((float(conf_and_target.split('\t')[0]), float(conf_and_target.split('\t')[1]))) for conf_and_target in
         conf_and_target_array])

    target_abnormal_idxs_flag = (file_[:, 1] == 1)
    target_normal_idxs_flag = (file_[:, 1] == 0)
 
    start = np.min(file_[:, 0])
    end = np.max(file_[:, 0])
    
    gap = (end-start) / 20000.
    end = end+gap

    auroc = 0.0
    fprTemp = 1.0
    tpr_list, fpr_list = list(), list()
    # for delta in np.arange(start, end, gap):
    for delta in np.arange(start, end, gap):
        
        tpr = np.sum(file_[target_abnormal_idxs_flag, 0] >= delta) / np.sum(target_abnormal_idxs_flag)
        fpr = np.sum(file_[target_normal_idxs_flag, 0] >= delta) / np.sum(target_normal_idxs_flag)
        # print(start, end, gap, delta, tpr, fpr)
        # exit()
        # f1.write("{}\n".format(tpr))
        # f2.write("{}\n".format(fpr))
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        auroc += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    
    fig, ax = plt.subplots()
    ax.plot(fpr_list, tpr_list,  linewidth=3 )
    ax.axhline(y=1.0, color='black', linestyle='dashed')
    ax.set_title('ROC {} {}epoch'.format(category, epoch))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    plt.xlabel('FPR(False Positive Rate)')
    plt.ylabel('TPR(True Positive Rate)')
    ax.text(0.6,0.1, s='auroc : {:.5f}'.format(auroc),  fontsize=15)

    if save_dir is None:    # load dir에 저장
        ax.figure.savefig('{}/ROC_{}_{}_{}epoch_{}.jpg'.format(load_dir, category, data_type, epoch, step), dpi=300)
    else :                  # 다른 dir을 지정하여 저장
        ax.figure.savefig('{}/ROC_{}_{}_{}epoch_{}.jpg'.format(save_dir, category, data_type, epoch, step), dpi=300)
    plt.close("all")
    return auroc

    

# roc_curve_plot('/home/lhj/code/medical_codes/Hemodialysis/result/rnn_v3/Classification/Untitled Folder 2/','map', 'Validation', 0)

# roc_curve_plot('/home/lhj/code/medical_codes/Hemodialysis/result/rnn_v3/Classification/Untitled Folder 2/','sbp', 'Validation', 0)

# roc_curve_plot('/home/lhj/code/medical_codes/Hemodialysis/result/rnn_v3/Classification/Untitled Folder 2/',
#     'under90', 'Validation', 0)
# exit()

