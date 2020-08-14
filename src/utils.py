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


# version3 용 eval.
def eval_rnn_classification_v3(loader, model, device, output_size, criterion, threshold=0.5, log_dir=None, epoch=None, step=0):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        total_output = torch.tensor([], dtype=torch.float).to(device)
        total_target = torch.tensor([]).to(device)

        for i, ((inputs_fix, inputs_seq), (targets), seq_len) in enumerate(loader):
            inputs_fix = inputs_fix.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = targets.float().to(device)
            output = model(inputs_fix, inputs_seq, seq_len, device) # shape : (seq, batch size, 5)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                flattened_output = torch.cat([flattened_output, output[:seq, idx, :].reshape(-1, output_size)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, output_size)), dim=0)

            loss_sbp = criterion(flattened_output[:,0], flattened_target[:,0])
            loss_map = criterion(flattened_output[:,1], flattened_target[:,1])
            loss_under90 = criterion(flattened_output[:,2], flattened_target[:,2])
            loss_sbp2 = criterion(flattened_output[:,3], flattened_target[:,3])
            loss_map2 = criterion(flattened_output[:,4], flattened_target[:,4])

            loss = loss_sbp + loss_map + loss_under90 + loss_sbp2 + loss_map2
            running_loss += loss.item()

            total_target = torch.cat([total_target, flattened_target], dim=0)
            total_output = torch.cat([total_output, flattened_output], dim=0)
        print("\tEvaluated Loss : {:.4f}".format(running_loss / i))
        result_dict = confidence_save_and_cal_auroc(F.sigmoid(total_output), total_target, 'Validation', log_dir, epoch, step)

        val_total = len(total_output)

        # for thres in threshold:
        #     val_correct0, val_correct1, val_correct2, val_correct3, val_correct4 = 0, 0, 0, 0, 0

        #     pred0 = (F.sigmoid(total_output[:,0]) > thres).long()
        #     pred1 = (F.sigmoid(total_output[:,1]) > thres).long()
        #     pred2 = (F.sigmoid(total_output[:,2]) > thres).long()
        #     pred3 = (F.sigmoid(total_output[:,3]) > thres).long()
        #     pred4 = (F.sigmoid(total_output[:,4]) > thres).long()

        #     val_correct0 += (pred0 == total_target[:,0].long()).sum().item()
        #     val_correct1 += (pred1 == total_target[:,1].long()).sum().item()
        #     val_correct2 += (pred2 == total_target[:,2].long()).sum().item()
        #     val_correct3 += (pred3 == total_target[:,3].long()).sum().item()
        #     val_correct4 += (pred4 == total_target[:,4].long()).sum().item()

        #     sbp_confusion_matrix, sbp_log = confusion_matrix(pred0, total_target[:,0], 2)
        #     map_confusion_matrix, dbp_log = confusion_matrix(pred1, total_target[:, 1], 2)
        #     under90_confusion_matrix, dbp_log = confusion_matrix(pred2, total_target[:, 2], 2)
        #     sbp2_confusion_matrix, dbp_log = confusion_matrix(pred3, total_target[:, 3], 2)
        #     map2_confusion_matrix, dbp_log = confusion_matrix(pred4, total_target[:, 4], 2)
        #     confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(),
        #                                        log_dir + '/{}'.format(thres),
        #                                        epoch, step, 'val', 'sbp', v3=True)
        #     confusion_matrix_save_as_img(map_confusion_matrix.detach().cpu().numpy(),
        #                                        log_dir + '/{}'.format(thres),
        #                                        epoch, step, 'val', 'map', v3=True)
        #     confusion_matrix_save_as_img(under90_confusion_matrix.detach().cpu().numpy(),
        #                                        log_dir + '/{}'.format(thres), epoch, step, 'val', 'under90', v3=True)
        #     confusion_matrix_save_as_img(sbp2_confusion_matrix.detach().cpu().numpy(),
        #                                        log_dir + '/{}'.format(thres), epoch, step, 'val', 'sbp2', v3=True)
        #     confusion_matrix_save_as_img(map2_confusion_matrix.detach().cpu().numpy(),
        #                                        log_dir + '/{}'.format(thres), epoch, step, 'val', 'map2', v3=True)


        #     print("\t Threshold: {} \tAccuracy of SBP: {:.2f}%\t MAP: {:.2f}%\t Under90: {:.2f}% \t SBP2: {:.2f}% \t MAP2: {:.2f}%".format(thres, 100 * val_correct0 / val_total,
        #                                                                                 100 * val_correct1 / val_total,
        #                                                                                 100 * val_correct2 / val_total,
        #                                                                                 100 * val_correct3 / val_total,
        #                                                                                 100 * val_correct4 / val_total))
            # print("\t Threshold: {} \tAccuracy of SBP: {:.2f}%\t MAP: {:.2f}%\t Under90: {:.2f}% \t".format(thres, 100 * val_correct0 / val_total, 100 * val_correct1 / val_total, 100 * val_correct2 / val_total))
        return result_dict

def roc_curve_plot_v2(load_dir, category, conf, target, epoch, save_dir=None):
    # calculate the AUROC
    from sklearn import metrics
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    average_precision = average_precision_score(target, conf)
    roc_fpr_array, roc_tpr_array, roc_thresholds = metrics.roc_curve(target, conf)
    pr_precision_array, pr_recall_array, pr_thresholds = metrics.precision_recall_curve(target, conf)
    auc = metrics.auc(roc_fpr_array, roc_tpr_array)

    fig = plt.figure(figsize=(10,5))
    if False:
        if category == 'under90':
            fig.suptitle("Under 90", fontsize=16)
        elif category == 'init_composite' :
            fig.suptitle("composite from init timeframe", fontsize=16)
        elif category == 'curr_composite' :
            fig.suptitle("composite from present timeframe", fontsize=16)
        else:
            print('subfigure sub title error')
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(roc_fpr_array, roc_tpr_array,  linewidth=3 )
    ax1.axhline(y=1.0, color='black', linestyle='dashed')
    # ax1.set_title('ROC {} {}epoch'.format(category, epoch))
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel('FPR(False Positive Rate)')
    ax1.set_ylabel('TPR(True Positive Rate)')
    ax1.text(0.5,0.05, s='auroc : {:.5f}'.format(auc),  fontsize=15)

    ax2 = fig.add_subplot(1,2,2)
    # ax2.plot(tpr_list, prec_list,  linewidth=3 )
    ax2.plot(pr_recall_array, pr_precision_array,  linewidth=3 )
    ax2.axhline(y=1.0, color='black', linestyle='dashed')
    # ax2.set_title('PR_Curve {} {}epoch'.format(category, epoch))
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel('TPR(Recall)')
    ax2.set_ylabel('Precision')
    ax2.text(0.05,0.05, s='AP : {:.5f}'.format(average_precision),  fontsize=15)


    if save_dir is None:  # load dir에 저장
        fig.savefig('{}/ROC_{}_{}.jpg'.format(load_dir, category, epoch), dpi=300)
        # fig.savefig('{}/ROC_{}_{}.png'.format(load_dir, category, epoch), dpi=300)
        # fig.savefig('{}/ROC_{}_{}.pdf'.format(load_dir, category, epoch), dpi=300)
    else:  # 다른 dir을 지정하여 저장
        with open('{}/{}_roc_tpr.txt'.format(save_dir, category), 'w') as f1, open('{}/{}_roc_fpr.txt'.format(save_dir, category), 'w') as f2, open('{}/{}_pr_prec.txt'.format(save_dir, category), 'w') as f3, open('{}/{}_pr_recall.txt'.format(save_dir, category), 'w') as f4 :
            if len(roc_tpr_array) != len(roc_fpr_array):
                print('length error / roc')
                assert len(roc_tpr_array) != len(roc_fpr_array)
            if len(pr_precision_array) != len(pr_recall_array):
                print('length error  / pr')
                assert len(pr_precision_array) != len(pr_recall_array)
            for i in range(len(roc_tpr_array)):
                f1.write('{}\n'.format(roc_tpr_array[i]))
                f2.write('{}\n'.format(roc_fpr_array[i]))
            for i in range(len(pr_precision_array)):
                f3.write('{}\n'.format(pr_precision_array[i]))
                f4.write('{}\n'.format(pr_recall_array[i]))
        fig.savefig('{}/ROC_{}_{}.jpg'.format(save_dir, category, epoch), dpi=300)
    plt.close("all")
    return auc, average_precision, [roc_tpr_array, roc_fpr_array, pr_precision_array, pr_recall_array]


def confidence_save_and_cal_auroc(mini_batch_outputs, mini_batch_targets, data_type, save_dir, epoch=9999, step=0):
    '''
    mini_batch_outputs shape : (data 개수, 3) -> flattend_output    / cf. data개수 --> 각 투석의 seq_len의 total sum
    data_type : Train / Validation / Test
    minibatch 별로 저장 될 수 있게 open(,'a')로 했는데, 저장 다 하면 f.close() 권하긴 함.
    KY: Batch 별로 작동되게 수정하여 f.close() 추가
    '''
    print("Making roc curve...")
    start_time = time.time()
    save_dir += '/auroc'
    category = {'IDH1':0, 'IDH2':1, 'IDH3':2, 'IDH4':3, 'IDH5':4}
    result_dict = dict()
    for key, value in category.items():
        key_dir = save_dir + '/' + key
        make_dir(key_dir)
        # f = open('{}/conf_{}_{}_{}_{}.txt'.format(key_dir, epoch, step, data_type, key), 'w')
        # for i in range(len(mini_batch_outputs[:,value])):
        #     f.write("{}\t{}\n".format(mini_batch_outputs[i,value].item(), mini_batch_targets[i,value].item()))
        # f.close()
        auroc, average_precision, [roc_tpr_array, roc_fpr_array, pr_precision_array, pr_recall_array] = roc_curve_plot_v2(load_dir=key_dir, category=key, conf=mini_batch_outputs[:,value].detach().cpu().numpy(), target=mini_batch_targets[:,value].detach().cpu().numpy(), epoch=epoch)
        print("{} s |  {}  auroc : {}  AP : {} ".format(time.time() - start_time , key, auroc, average_precision))
        result_dict[key] = [auroc, average_precision]
    
    return result_dict