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
from sys import exit

input_fix_size = 109
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

os.chdir('/home/ky/Desktop/Project/의료/result/rnn_v3/Classification/Dec12_171625/src')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = RNN_V3(input_fix_size, input_seq_size, hidden_size, num_layers, output_size, batch_size, dropout_rate,num_class1, num_class2, num_class3).to(device)
state = torch.load('/home/ky/Desktop/Project/의료/result/rnn_v3/Classification/Dec12_171625/bs32_lr0.01_wdecay5e-06/12epoch.model')
model.load_state_dict(state['model'])
model.to(device)
model.eval()

train_data = torch.load('/home/ky/Desktop/Project/의료/data/tensor_data/1210_EF_60min/Train.pt')
val_data = torch.load('/home/ky/Desktop/Project/의료/data/tensor_data/1210_EF_60min/Validation.pt')
# val_data = val_data[:int(len(val_data) * 0.1)]
full_idx = [i for i in range(len(train_data[0][0]))]
# seq_idx = [5, 6, 11, 12, 13, 14, 15, 16] + [i for i in range(len(train_data[0][0]) - 11, len(train_data[0][0]) - 1)]
seq_idx = [5, 6, 11, 12, 13, 14, 15, 16] + [i + len(full_idx) for i in range(-10, 0)]  # add HD_ntime_target
fix_idx = [i for i in full_idx if i not in seq_idx]

val_data_ = []
for i in range(len(val_data)):
    val_data_.append([val_data[i][0, fix_idx], val_data[i][:, seq_idx]])
val_data = val_data_
del val_data_

val_seq_len_list = [len(x[1]) for x in val_data]
val_dataset = loader.RNN_Dataset((val_data, val_seq_len_list), type='cls', ntime=60)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False,
                        collate_fn=lambda batch: loader.pad_collate(batch, True))

model.eval()
BCE_loss_with_logit = nn.BCEWithLogitsLoss().to(device)

# For Control only : only masking latter half
with torch.no_grad():
    total_output = torch.tensor([], dtype=torch.float).to(device)
    total_target = torch.tensor([]).to(device)

    for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(val_loader):
        feature_len = inputs_seq.shape[-1]
        inputs_fix = inputs_fix.to(device)
        inputs_seq = inputs_seq.to(device)
        targets = targets_real.float().to(device)
        output = model(inputs_fix, inputs_seq, seq_len, device)  # shape : (seq, batch size, 5)

        cut_idx = [i// 2 for i in seq_len]

        flattened_output = torch.tensor([]).to(device)
        flattened_target = torch.tensor([]).to(device)
        for idx, seq in enumerate(seq_len):
            flattened_output = torch.cat([flattened_output, output[:seq, idx, :].reshape(-1, 5)], dim=0)
            flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, 5)), dim=0)

        total_target = torch.cat([total_target, flattened_target], dim=0)
        total_output = torch.cat([total_output, flattened_output], dim=0)
    # print("\t[TIMER] INTO CONFIDENCE SCORE... {} // {}".format(time.time() - intermediate, time.time() - start))
    # intermediate = time.time()
    # utils.confidence_save_and_cal_auroc(F.sigmoid(total_co_output), total_target, 'Validation', 'control_corrupted', 0, 0, cal_roc=True)
    utils.confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', 'Noise/Original', 0, 0, cal_roc=True)


with torch.no_grad():
    total_output = torch.tensor([], dtype=torch.float).to(device)
    total_co_output = torch.tensor([], dtype=torch.float).to(device)
    total_target = torch.tensor([]).to(device)

    for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(val_loader):
        mask = torch.ones((inputs_seq.shape[:2]))
        feature_len = inputs_seq.shape[-1]
        inputs_fix = inputs_fix.to(device)
        inputs_seq = inputs_seq.to(device)
        corrupted_inputs = inputs_seq.clone()

        targets = targets_real.float().to(device)
        cut_idx = [i// 2 for i in seq_len]

        for idx, j in enumerate(cut_idx):
            noise = np.random.uniform(-5, 5, size=(j, feature_len))
            corrupted_inputs[:j, idx, :] = torch.tensor(noise).to(device)
            # noise = np.random.uniform(-0.4, 0.4, size=(inputs_fix.shape[0], feature_len))
            # corrupted_inputs += torch.tensor(noise).to(device)
            mask[:j, idx] = 0

        mask = mask.byte().unsqueeze(-1).to(device)
        targets = torch.masked_select(targets, mask).view(-1, 5)

        output = model(inputs_fix, inputs_seq, seq_len, device)  # shape : (seq, batch size, 5)
        output = torch.masked_select(output, mask).view(-1,5)
        corrupted_output = model(inputs_fix, corrupted_inputs, seq_len, device)
        # corrupted_output = model(corrupted_inputs, inputs_seq, seq_len, device)
        corrupted_output = torch.masked_select(corrupted_output, mask).view(-1, 5)

        total_target = torch.cat([total_target, targets], dim=0)
        total_co_output = torch.cat([total_co_output, corrupted_output], dim=0)
        total_output = torch.cat([total_output, output], dim=0)

    # print("\t[TIMER] INTO CONFIDENCE SCORE... {} // {}".format(time.time() - intermediate, time.time() - start))
    # intermediate = time.time()
    utils.confidence_save_and_cal_auroc(F.sigmoid(total_co_output), total_target, 'Validation', '_Noise/corrupted', 0,
                                        0, cal_roc=True)
    utils.confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', '_Noise/control', 0, 0,
                                        cal_roc=True)


# Sample only one sequence and add noise on current or previous
np.random.seed(1230)

with torch.no_grad():
    total_output = torch.tensor([], dtype=torch.float).to(device)
    total_co_output = torch.tensor([], dtype=torch.float).to(device)
    total_target = torch.tensor([]).to(device)

    for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(val_loader):
        feature_len = inputs_seq.shape[-1]

        mask = torch.zeros((inputs_seq.shape[:2]))
        inputs_fix = inputs_fix.to(device)
        inputs_seq = inputs_seq.to(device)
        corrupted_inputs = inputs_seq.clone()

        targets = targets_real.float().to(device)
        cut_idx = [i// 2 for i in seq_len]

        for idx, j in enumerate(seq_len):
            if j == 1:
                continue
            sampled_seq = np.random.randint(1,j)
            # print(sampled_seq)
            noise = np.random.uniform(-5, 5, size=(feature_len))
            corrupted_inputs[sampled_seq-1, idx, :] = torch.tensor(noise).to(device)
            mask[sampled_seq, idx] = 1

        mask = mask.byte().unsqueeze(-1).to(device)
        targets = torch.masked_select(targets, mask).view(-1, 5)

        output = model(inputs_fix, inputs_seq, seq_len, device)  # shape : (seq, batch size, 5)
        output = torch.masked_select(output, mask).view(-1,5)
        corrupted_output = model(inputs_fix, corrupted_inputs, seq_len, device)
        corrupted_output = torch.masked_select(corrupted_output, mask).view(-1, 5)

        total_target = torch.cat([total_target, targets], dim=0)
        total_co_output = torch.cat([total_co_output, corrupted_output], dim=0)
        total_output = torch.cat([total_output, output], dim=0)

    # print("\t[TIMER] INTO CONFIDENCE SCORE... {} // {}".format(time.time() - intermediate, time.time() - start))
    # intermediate = time.time()
    utils.confidence_save_and_cal_auroc(F.sigmoid(total_co_output), total_target, 'Validation', '10Noise_at_sampled/current_corrupted', 0,
                                        0, cal_roc=True)
    utils.confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', '10Noise_at_sampled/current_control', 0, 0,
                                        cal_roc=True)
