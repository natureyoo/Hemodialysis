import torch
from models import *
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
import loader
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
threshold = [0.1, 0.3, 0.5, 0.7]
epoch = 11
step = 0
task_type = 'Classification'
log_dir = 'Eval_on_Test'

# os.chdir('/home/ky/Desktop/Project/의료/result/rnn_v3/Classification/Dec12_171625/src')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = RNN_V3(input_fix_size, input_seq_size, hidden_size, num_layers, output_size, batch_size, dropout_rate,num_class1, num_class2, num_class3).to(device)
state = torch.load('/home/ky/Desktop/Project/의료/result/rnn_v3/Classification/Dec18_161630/bs32_lr0.01_wdecay5e-06/11epoch.model')
model.load_state_dict(state['model'])
model.to(device)
model.eval()

val_data = torch.load('/home/ky/Desktop/Project/의료/data/tensor_data/1218_EF_60min/Test.pt')
# val_data = val_data[:int(len(val_data) * 0.1)]
full_idx = [i for i in range(len(val_data[0][0]))]
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
    utils.confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Test', log_dir, epoch, step, cal_roc=True)

    val_total = len(total_output)

    for thres in threshold:
        print("*****THRESHOLD: {}*****".format(thres))
        pred0 = (F.sigmoid(total_output[:,0]) > thres).long()
        print("SBP:")
        utils.compute_f1score(total_target[:,0], pred0, True)
        pred1 = (F.sigmoid(total_output[:,1]) > thres).long()
        print("MAP:")
        utils.compute_f1score(total_target[:,1], pred1, True)
        pred2 = (F.sigmoid(total_output[:,2]) > thres).long()
        print("Under90:")
        utils.compute_f1score(total_target[:,2], pred2, True)
        pred3 = (F.sigmoid(total_output[:,3]) > thres).long()
        print("SBP2:")
        utils.compute_f1score(total_target[:,3], pred3, True)
        pred4 = (F.sigmoid(total_output[:,4]) > thres).long()
        print("MAP2:")
        utils.compute_f1score(total_target[:,4], pred4, True)

        sbp_confusion_matrix, sbp_log = utils.confusion_matrix(pred0, total_target[:,0], 2)
        map_confusion_matrix, dbp_log = utils.confusion_matrix(pred1, total_target[:, 1], 2)
        under90_confusion_matrix, dbp_log = utils.confusion_matrix(pred2, total_target[:, 2], 2)
        sbp2_confusion_matrix, dbp_log = utils.confusion_matrix(pred3, total_target[:, 3], 2)
        map2_confusion_matrix, dbp_log = utils.confusion_matrix(pred4, total_target[:, 4], 2)

        utils.confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(),
                                           log_dir + '/{}'.format(thres),
                                           epoch, step, 'Test', 'SBP', v3=True)
        utils.confusion_matrix_save_as_img(map_confusion_matrix.detach().cpu().numpy(),
                                           log_dir + '/{}'.format(thres),
                                           epoch, step, 'Test', 'MAP', v3=True)
        utils.confusion_matrix_save_as_img(under90_confusion_matrix.detach().cpu().numpy(),
                                           log_dir + '/{}'.format(thres), epoch, step, 'Test', 'Under90', v3=True)
        utils.confusion_matrix_save_as_img(sbp2_confusion_matrix.detach().cpu().numpy(),
                                           log_dir + '/{}'.format(thres), epoch, step, 'Test', 'SBP2', v3=True)
        utils.confusion_matrix_save_as_img(map2_confusion_matrix.detach().cpu().numpy(),
                                           log_dir + '/{}'.format(thres), epoch, step, 'Test', 'MAP2', v3=True)
