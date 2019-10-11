import pandas as pd
import numpy as np
import time
import glob
from models import *
import torch.nn as nn
import sklearn.metrics
import os
import numpy as np

targets_list = ['VS_sbp_target', 'VS_dbp_target', 'clas_target']
id_features = ['ID_hd', 'ID_timeline', 'ID_class']

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

def eval_regression(model, loader, device, log_dir, save_result=False, criterion=nn.L1Loss(reduction='sum')):
    if save_result:
        f = open("{}/test_result.csv".format(log_dir), 'ab')
    with torch.no_grad():
        model.eval()
        running_loss = 0
        total = 0
        for (inputs, targets) in loader:
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            targets = targets.float().view(-1, 1)
            if save_result:
                concat = torch.stack((outputs, targets), dim=1).squeeze(dim=-1)
                np.savetxt(f, concat.data.cpu().numpy())

            val_loss = criterion(outputs, targets)
            total += inputs.size(0)
            running_loss += val_loss.item()
        print("   L1 loss on Validation: {:.4f}".format(running_loss / total))
    if save_result:
        f.close()
    return running_loss, total

def eval_rnn(loader, model, device, output_size, criterion=nn.L1Loss(reduction="sum")):

    print("********Start Evaluating ********")
    total = 0
    running_loss = 0

    for i, (x, y, seq_len) in enumerate(loader):
        # batch_size = y.size(0)
        x = x.permute(1,0,2).float().to(device)
        y = y.float().to(device)
        seq_len = seq_len.to(device)
        # print('Shape of x : ', x.shape)

        # Forward pass
        outputs = model(x, seq_len, device)
        # Initialize
        flattened_y = y[0,:seq_len[0]].view(-1,output_size)
        # print("flattened_y", flattened_y.size())
        flattened_output = outputs[0,:seq_len[0],:].view(-1,output_size)
        # print("flattened_output", flattened_output.size())

        for idx,seq in enumerate(seq_len[1:]):
            # print("Output of single batch:", outputs[idx+1,:seq,:].size())
            flattened_output = torch.cat([flattened_output,outputs[idx+1,:seq,:].view(-1,output_size)], dim=0)
            flattened_y = torch.cat((flattened_y,y[idx+1,:seq].view(-1,output_size)), dim=0)

        loss = criterion(flattened_y, flattened_output)
        total += seq_len.sum().item()
        running_loss += loss
    print("Validation Loss : {:.4f} \n".format(running_loss/total))

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