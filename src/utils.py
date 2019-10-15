import pandas as pd
import time
import glob
from models import *
import torch.nn as nn
import sklearn.metrics
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt

targets_list = ['VS_sbp_target', 'VS_dbp_target', 'clas_target']
id_features = ['ID_hd', 'ID_timeline', 'ID_class']

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

def save_snapshot(network, optimizer, snapshot_dir, epoch, iteration, snapshot_epoch_fre):
    if epoch % snapshot_epoch_fre == 0:
        dir_name = 'snapshot/epoch-%04d' % epoch
        save_dir = os.path.join(snapshot_dir, dir_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        network_path = os.path.join(save_dir, 'model.pth')
        torch.save(network.state_dict(), network_path)
        optimizer_path = os.path.join(save_dir, 'optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
        # print('[SAVE] {} {}]' .format(network_path, optimizer_path))

def save_result_txt(outputs, targets, save_root, epoch, Flag='Test', ID=None, initial=None):
    txt_save_root = os.path.join(save_root + '/result/')
    if not os.path.isdir(txt_save_root):
        os.makedirs(txt_save_root)

    with open(txt_save_root + 'result_{}epoch_{}.txt'.format(epoch, Flag), 'a') as f:
        if ID is not None:
            f.write('ID : {}\n'.format(str(ID.data.cpu().numpy())))
        f.write('{:25}  {:25}\n'.format('target', 'pred'))
        if initial is not None:
            f.write('{:25}  <<-- initial [sbp,dbp]\n'.format(str(initial.int().data.cpu().numpy())))

        for output, target in zip(outputs, targets):
            f.write('{:25}  {:25}'.format(str(target.int().data.cpu().numpy()), str(output.int().data.cpu().numpy())))
            f.write('\n')
        f.write('\n')

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

def eval_rnn(loader, model, device, output_size, criterion=nn.L1Loss(reduction="sum")):
    with torch.no_grad():
        running_loss = 0
        total = 0

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

            loss = criterion(flattened_target, flattened_output)
            total += len(seq_len)
            running_loss += loss.item()

        print("Evaluated Loss : {:.4f} \n".format(running_loss/total), end=' ')

    return running_loss/total, total

def confusion_matrix(preds, labels, n_classes):

    conf_matrix = torch.zeros(n_classes, n_classes)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1

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

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))
        print('Sensitivity = {:.4f}'.format(sensitivity))
        print('Specificity = {:.4f}'.format(specificity))
        print('\n')

    return conf_matrix, (sensitivity_log, specificity_log)

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
    model.load_state_dict(state['model'])
    optimizer = optimizer(model.parameters())
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
