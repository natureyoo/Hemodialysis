import torch
from models import *
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
import loader
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import os
import pandas as pd

# input_size = 36
# input_size = 143
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
    ax.set_title('{} Confusion Matrix'.format(name))
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label')
    plt.xlabel('Pred. label')

    ax.figure.savefig('{}/{}_{}.jpg'.format(save_dir, data_type, name))
    # ax.figure.savefig('{}/{}_{}_{}epoch_{}iter_count.jpg'.format(save_dir, data_type, name, epoch, iteration))
    plt.close("all")


    matrix_sum = matrix.sum(axis=1)
    for i in range(len(matrix)):
        matrix[i] /= matrix_sum[i]

    df_cm = pd.DataFrame(matrix, index = [str(i) for i in range(num_class)], columns = [str(i) for i in range(num_class)])
    plt.figure(figsize = (9,7))
    ax = sn.heatmap(df_cm, annot=True, cmap='RdBu_r', vmin=0, vmax=1, annot_kws={"size": 15})
    ax.set_title('{} Confusion Matrix'.format(name))
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label')
    plt.xlabel('Pred. label')

    ax.figure.savefig('{}/{}_{}.jpg'.format(save_dir, data_type, name))
    # ax.figure.savefig('{}/{}_{}_{}epoch_{}iter.jpg'.format(save_dir, data_type, name, epoch, iteration))
    plt.close("all")

def confidence_save_and_cal_auroc(mini_batch_outputs, mini_batch_targets, data_type, save_dir, epoch=9999, step=0, composite=True, cal_roc=True):
    '''
    mini_batch_outputs shape : (data 개수, 3) -> flattend_output    / cf. data개수 --> 각 투석의 seq_len의 total sum
    data_type : Train / Validation / Test
    minibatch 별로 저장 될 수 있게 open(,'a')로 했는데, 저장 다 하면 f.close() 권하긴 함.
    KY: Batch 별로 작동되게 수정하여 f.close() 추가
    '''
    print("Making roc curve...")
    save_dir += '/auroc'
    category ={"Composite Outcome from Initial Timeframe":0, "Composite Outcome from Present Timeframe":1}

    for key, value in category.items():
        key_dir = save_dir + '/' + key
        if not os.path.isdir(key_dir):
            os.makedirs(key_dir)
        f = open('{}/confidence_{}_{}.txt'.format(key_dir, data_type, key), 'w')
        for i in range(len(mini_batch_outputs[:,value])):
            f.write("{}\t{}\n".format(mini_batch_outputs[i,value].item(), mini_batch_targets[i,value].item()))
        f.close()
        if cal_roc :
            auroc = roc_curve_plot(key_dir, key, data_type, epoch, step)

def roc_curve_plot(load_dir, category='sbp', data_type='Test', epoch=None, step=0, save_dir=None):
    # calculate the AUROC
    # f1 = open('{}/Update_tpr.txt'.format(load_dir), 'w')
    # f2 = open('{}/Update_fpr.txt'.format(load_dir), 'w')

    conf_and_target_array = np.loadtxt(
        '{}/confidence_{}_{}.txt'.format(load_dir, data_type, category),
        delimiter=',', dtype=np.str)
    file_ = np.array(
        [np.array((float(conf_and_target.split('\t')[0]), float(conf_and_target.split('\t')[1]))) for conf_and_target in
         conf_and_target_array])

    target_abnormal_idxs_flag = (file_[:, 1] == 1)
    target_normal_idxs_flag = (file_[:, 1] == 0)

    start = np.min(file_[:, 0])
    end = np.max(file_[:, 0])

    gap = (end - start) / 20000.
    end = end + gap

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
    ax.plot(fpr_list, tpr_list, linewidth=3)
    ax.axhline(y=1.0, color='black', linestyle='dashed')
    ax.set_title('ROC of {}'.format(category, epoch))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    plt.xlabel('FPR(False Positive Rate)')
    plt.ylabel('TPR(True Positive Rate)')
    ax.text(0.6, 0.1, s='auroc : {:.5f}'.format(auroc), fontsize=15)

    if save_dir is None:  # load dir에 저장
        ax.figure.savefig('{}/ROC_{}_{}.jpg'.format(load_dir, category, data_type), dpi=300)
    else:  # 다른 dir을 지정하여 저장
        ax.figure.savefig('{}/ROC_{}_{}.jpg'.format(save_dir, category, data_type), dpi=300)
    plt.close("all")
    return auroc


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = RNN_V3(input_fix_size, input_seq_size, hidden_size, num_layers, output_size, batch_size, dropout_rate,num_class1, num_class2, num_class3).to(device)
state = torch.load('/home/ky/Desktop/Project/의료/final_result/rnn_v3/Classification/Dec30_101341/bs32_lr0.01_wdecay5e-06/12epoch_fix_input_size_110.model')
model.load_state_dict(state['model'])
model.to(device)
model.eval()

val_data = torch.load('../tensor_data/1230_RNN_60min/Test.pt')
# val_data = val_data[:int(len(val_data) * 0.01)]
full_idx = [i for i in range(len(val_data[0][0]))]
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

BCE_loss_with_logit = nn.BCEWithLogitsLoss().to(device)
log_dir = 'Composite/'
with torch.no_grad():
    running_loss = 0
    total_output = torch.tensor([], dtype=torch.float).to(device)
    total_target = torch.tensor([]).to(device)

    for i, ((inputs_fix, inputs_seq), (targets, targets_real), seq_len) in enumerate(val_loader):
        inputs_fix = inputs_fix.to(device)
        inputs_seq = inputs_seq.to(device)
        targets = targets_real.float().to(device)

        Initial_composite_targets, _ = torch.max(targets_real[:,:,:2], 2, keepdim=True)
        Current_composite_targets, _ = torch.max(targets_real[:,:,3:], 2, keepdim=True)
        output = model(inputs_fix, inputs_seq, seq_len, device)  # shape : (seq, batch size, 5)
        Initial_composite_output, _ = torch.max(output[:,:,:2], 2, keepdim=True)
        Current_composite_output, _ = torch.max(output[:,:,3:], 2, keepdim=True)

        # OR_output = torch.mean(output[:, :, :], 2, keepdim=True)
        Initial_composite_targets = Initial_composite_targets.float().to(device)
        Current_composite_targets = Current_composite_targets.float().to(device)
        Initial_composite_output = Initial_composite_output.float().to(device)
        Current_composite_output = Current_composite_output.float().to(device)

        flattened_output = torch.tensor([]).to(device)
        flattened_target = torch.tensor([]).to(device)

        for idx, seq in enumerate(seq_len):
            tmp_output = torch.cat([Initial_composite_output[:seq, idx, :].reshape(-1, 1), Current_composite_output[:seq, idx, :].reshape(-1, 1)], dim=1)
            tmp_target = torch.cat([Initial_composite_targets[:seq, idx, :].reshape(-1, 1), Current_composite_targets[:seq, idx, :].reshape(-1, 1)], dim=1)
            flattened_output = torch.cat([flattened_output, tmp_output], dim=0)
            flattened_target = torch.cat([flattened_target, tmp_target], dim=0)


        total_target = torch.cat([total_target, flattened_target], dim=0)
        total_output = torch.cat([total_output, flattened_output], dim=0)
    confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Test', log_dir, 0, 0,
                                  cal_roc=True)

    val_total = len(total_output)

    for thres in [0.1,0.3,0.5,0.7]:
        pred0 = (F.sigmoid(total_output[:,0]) > thres).long()
        pred1 = (F.sigmoid(total_output[:,1]) > thres).long()

        initial_confusion_matrix, log = utils.confusion_matrix(pred0, total_target[:,0], 2)
        current_confusion_matrix, log = utils.confusion_matrix(pred1, total_target[:,1], 2)

        confusion_matrix_save_as_img(initial_confusion_matrix.detach().cpu().numpy(),
                                     log_dir + '/{}'.format(thres),
                                     0, 0, 'Test', 'Composite Outcome from Initial Timeframe', v3=True)
        confusion_matrix_save_as_img(current_confusion_matrix.detach().cpu().numpy(),
                                     log_dir + '/{}'.format(thres),
                                     0, 0, 'Test', 'Composite Outcome from Present Timeframe', v3=True)
