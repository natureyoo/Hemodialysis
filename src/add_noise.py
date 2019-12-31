import torch
from models import *
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
import loader
import utils
import argparse
from datetime import datetime
import sys
import utils
import os
import numpy as np
from sys import exit
import matplotlib.pyplot as plt

input_fix_size = 110
input_seq_size = 9
hidden_size = 256
num_layers = 3
num_epochs = 100
output_size = 5
num_class1 = 1
num_class2 = 1
num_class3 = 1
batch_size = 32
dropout_rate = 0.1
learning_rate = 0.01
w_decay = 0.1
task_type = 'Classification'

# os.chdir('/home/ky/Desktop/Project/의료/result/rnn_v3/Classification/Dec12_171625/src')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = RNN_V3(input_fix_size, input_seq_size, hidden_size, num_layers, output_size, batch_size, dropout_rate,num_class1, num_class2, num_class3).to(device)
state = torch.load('/home/jayeon/Documents/12epoch_fix_input_size_110.model')
model.load_state_dict(state['model'])
model.to(device)
model.eval()

# train_data = torch.load('/home/ky/Desktop/Project/의료/data/tensor_data/1210_EF_60min/Train.pt')
val_data = torch.load('/home/jayeon/Documents/code/Hemodialysis/data/tensor_data/1230_RNN_60min/Test.pt')
full_idx = [i for i in range(len(val_data[0][0]))]
seq_idx = [5, 6, 11, 12, 13, 14, 15, 16] + [i + len(full_idx) for i in range(-10, 0)]  # add HD_ntime_target
fix_idx = [i for i in full_idx if i not in seq_idx]

def add_noise_data(data, feat_idx=None):
    data_ = []
    for i in range(len(data)):
        if feat_idx is not None:
            data[i][:, feat_idx] += np.random.normal(0, 1, size=data[i][:, feat_idx].shape)
        data_.append([data[i][0, fix_idx], data[i][:, seq_idx]])
    return data_

for noise_idx in range(114, 119): #range(-1, input_fix_size + input_seq_size):
    val_data = torch.load('/home/jayeon/Documents/code/Hemodialysis/data/tensor_data/1230_RNN_60min/Test.pt')
    if noise_idx == -1:
        noise_idx = None
    val_data = add_noise_data(val_data, noise_idx)

    val_seq_len_list = [len(x[1]) for x in val_data]
    val_dataset = loader.RNN_Dataset((val_data, val_seq_len_list), type='cls', ntime=60)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False,
                            collate_fn=lambda batch: loader.pad_collate(batch, True))

    # For Control only : only masking latter half
    with torch.no_grad():
        total_output = torch.tensor([], dtype=torch.float).to(device)
        total_target = torch.tensor([]).to(device)

        for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(val_loader):
            inputs_fix = inputs_fix.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = targets_real.float().to(device)
            output = model(inputs_fix, inputs_seq, seq_len, device)  # shape : (seq, batch size, 5)

            Initial_composite_targets, _ = torch.max(targets_real[:, :, :2], 2, keepdim=True)
            Current_composite_targets, _ = torch.max(targets_real[:, :, 3:], 2, keepdim=True)

            Initial_composite_output, _ = torch.max(output[:, :, :2], 2, keepdim=True)
            Current_composite_output, _ = torch.max(output[:, :, 3:], 2, keepdim=True)

            Initial_composite_targets = Initial_composite_targets.float().to(device)
            Current_composite_targets = Current_composite_targets.float().to(device)
            Under90_targets = targets_real[:,:,2].float().to(device)
            Initial_composite_output = Initial_composite_output.float().to(device)
            Current_composite_output = Current_composite_output.float().to(device)
            Under90_output = output[:,:,2].float().to(device)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)
            for idx, seq in enumerate(seq_len):
                tmp_output = torch.cat([Initial_composite_output[:seq, idx, :].reshape(-1, 1), Current_composite_output[:seq, idx, :].reshape(-1, 1), Under90_output[:seq, idx].unsqueeze(1)], dim=1)
                tmp_target = torch.cat([Initial_composite_targets[:seq, idx, :].reshape(-1, 1), Current_composite_targets[:seq, idx, :].reshape(-1, 1), Under90_targets[:seq, idx].unsqueeze(1)], dim=1)
                flattened_output = torch.cat([flattened_output, tmp_output], dim=0)
                flattened_target = torch.cat([flattened_target, tmp_target], dim=0)

            total_target = torch.cat([total_target, flattened_target], dim=0)
            total_output = torch.cat([total_output, flattened_output], dim=0)

        print(total_target.shape)
        print(total_output.shape)
        utils.confidence_save_and_cal_auroc_for_niose(F.sigmoid(total_output), total_target, 'Test', 'Noise', noise_idx, cal_roc=True)

def plot_feature_importance(base_dir, columns):
    def plot_graph(columns, auroc, target_type, col_type=None):
        plt.rcdefaults()

        if col_type is None:
            fig, ax = plt.subplots(figsize=(20,40))
        else:
            fig, ax = plt.subplots(figsize=(10,15))
        y_pos = np.arange(len(columns))

        ax.barh(y_pos, -auroc, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(columns)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Loss of AUROC')
        if col_type is None:
            ax.set_title('Feature Importance on {}'.format(target_type))
        else:
            ax.set_title('{} Feature Importance on {}'.format(col_type, target_type))

        if col_type is None:
            plt.savefig(os.path.join(base_dir, 'feature_importance_on_{}.jpg'.format(target_type)))
        else:
            plt.savefig(os.path.join(base_dir, '{}_feature_importance_on_{}.jpg'.format(col_type, target_type)))
        plt.close()

    for target_type in ['init', 'curr', 'under90']:
        dir = os.path.join(base_dir, 'auroc', target_type)
        auroc_arr = np.zeros(len(columns), dtype=float)
        with open(os.path.join(dir, 'auroc_all.txt'), 'r') as f:
            for row in f.readlines():
                if row.split('\t')[0] =='original':
                    original_auroc = float(row.split('\t')[1])
                else:
                    auroc_arr[int(row.split('\t')[0].split('_')[1])] = float(row.split('\t')[1])

        auroc_arr -= original_auroc
        sorted_columns = []
        for idx in np.argsort(auroc_arr):
            sorted_columns.append(columns[idx])
        sorted_auroc_arr = np.sort(auroc_arr)

        plot_graph(sorted_columns, sorted_auroc_arr, target_type)

        numerical_columns = []
        numerical_auroc = []
        for idx, col in enumerate(sorted_columns):
            if col in ['EF', 'Pt_age', 'HD_duration', 'HD_ntime', 'HD_ctime', 'HD_prewt', 'HD_uf', 'VS_sbp', 'VS_dbp', 'VS_hr', 'VS_bt', 'VS_bfr', 'VS_uft', 'Lab_wbc', 'Lab_hb', 'Lab_plt', 'Lab_chol', 'Lab_alb', 'Lab_glu', 'Lab_ca', 'Lab_phos', 'Lab_ua', 'Lab_bun', 'Lab_scr', 'Lab_na', 'Lab_k', 'Lab_cl', 'Lab_co2']:
                numerical_columns.append(col)
                numerical_auroc.append(sorted_auroc_arr[idx])

        plot_graph(numerical_columns, np.array(numerical_auroc), target_type, 'numerical')


def run_plot():
    base_dir = '/home/jayeon/Documents/code/Hemodialysis/src/Noise2'
    columns = []
    with open('/home/jayeon/Documents/code/Hemodialysis/data/tensor_data/1230_RNN_60min/columns.csv', 'r') as f:
        for row in f.readlines():
            if row.strip() not in ['VS_sbp_target','VS_dbp_target','VS_sbp_target_class','VS_dbp_target_class']:
                columns.append(row.strip())

    plot_feature_importance(base_dir, columns)


# with torch.no_grad():
#     total_output = torch.tensor([], dtype=torch.float).to(device)
#     total_co_output = torch.tensor([], dtype=torch.float).to(device)
#     total_target = torch.tensor([]).to(device)
#
#     for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(val_loader):
#         mask = torch.ones((inputs_seq.shape[:2]))
#         feature_len = inputs_seq.shape[-1]
#         inputs_fix = inputs_fix.to(device)
#         inputs_seq = inputs_seq.to(device)
#         corrupted_inputs = inputs_seq.clone()
#
#         targets = targets_real.float().to(device)
#         cut_idx = [i// 2 for i in seq_len]
#
#         for idx, j in enumerate(cut_idx):
#             noise = np.random.uniform(-5, 5, size=(j, feature_len))
#             corrupted_inputs[:j, idx, :] = torch.tensor(noise).to(device)
#             # noise = np.random.uniform(-0.4, 0.4, size=(inputs_fix.shape[0], feature_len))
#             # corrupted_inputs += torch.tensor(noise).to(device)
#             mask[:j, idx] = 0
#
#         mask = mask.byte().unsqueeze(-1).to(device)
#         targets = torch.masked_select(targets, mask).view(-1, 5)
#
#         output = model(inputs_fix, inputs_seq, seq_len, device)  # shape : (seq, batch size, 5)
#         output = torch.masked_select(output, mask).view(-1,5)
#         corrupted_output = model(inputs_fix, corrupted_inputs, seq_len, device)
#         # corrupted_output = model(corrupted_inputs, inputs_seq, seq_len, device)
#         corrupted_output = torch.masked_select(corrupted_output, mask).view(-1, 5)
#
#         total_target = torch.cat([total_target, targets], dim=0)
#         total_co_output = torch.cat([total_co_output, corrupted_output], dim=0)
#         total_output = torch.cat([total_output, output], dim=0)
#
#     # print("\t[TIMER] INTO CONFIDENCE SCORE... {} // {}".format(time.time() - intermediate, time.time() - start))
#     # intermediate = time.time()
#     utils.confidence_save_and_cal_auroc(F.sigmoid(total_co_output), total_target, 'Validation', '_Noise/corrupted', 0,
#                                         0, cal_roc=True)
#     utils.confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', '_Noise/control', 0, 0,
#                                         cal_roc=True)
#
#
# # Sample only one sequence and add noise on current or previous
# np.random.seed(1230)
#
# with torch.no_grad():
#     total_output = torch.tensor([], dtype=torch.float).to(device)
#     total_co_output = torch.tensor([], dtype=torch.float).to(device)
#     total_target = torch.tensor([]).to(device)
#
#     for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(val_loader):
#         feature_len = inputs_seq.shape[-1]
#
#         mask = torch.zeros((inputs_seq.shape[:2]))
#         inputs_fix = inputs_fix.to(device)
#         inputs_seq = inputs_seq.to(device)
#         corrupted_inputs = inputs_seq.clone()
#
#         targets = targets_real.float().to(device)
#         cut_idx = [i// 2 for i in seq_len]
#
#         for idx, j in enumerate(seq_len):
#             if j == 1:
#                 continue
#             sampled_seq = np.random.randint(1,j)
#             # print(sampled_seq)
#             noise = np.random.uniform(-5, 5, size=(feature_len))
#             corrupted_inputs[sampled_seq-1, idx, :] = torch.tensor(noise).to(device)
#             mask[sampled_seq, idx] = 1
#
#         mask = mask.byte().unsqueeze(-1).to(device)
#         targets = torch.masked_select(targets, mask).view(-1, 5)
#
#         output = model(inputs_fix, inputs_seq, seq_len, device)  # shape : (seq, batch size, 5)
#         output = torch.masked_select(output, mask).view(-1,5)
#         corrupted_output = model(inputs_fix, corrupted_inputs, seq_len, device)
#         corrupted_output = torch.masked_select(corrupted_output, mask).view(-1, 5)
#
#         total_target = torch.cat([total_target, targets], dim=0)
#         total_co_output = torch.cat([total_co_output, corrupted_output], dim=0)
#         total_output = torch.cat([total_output, output], dim=0)
#
#     # print("\t[TIMER] INTO CONFIDENCE SCORE... {} // {}".format(time.time() - intermediate, time.time() - start))
#     # intermediate = time.time()
#     utils.confidence_save_and_cal_auroc(F.sigmoid(total_co_output), total_target, 'Validation', '10Noise_at_sampled/current_corrupted', 0,
#                                         0, cal_roc=True)
#     utils.confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', '10Noise_at_sampled/current_control', 0, 0,
#                                         cal_roc=True)
