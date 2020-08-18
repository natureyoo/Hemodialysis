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



def confusion_matrix_save_as_img(matrix, save_dir, epoch=0, data_type='train', name=None):
    mpl.use('Agg')
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

    make_dir(save_dir)
    ax.figure.savefig('{}/{}_{}epo_Count.jpg'.format(save_dir, name, epoch))
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

    ax.figure.savefig('{}/{}_{}epo.jpg'.format(save_dir, name, epoch))
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




def sklearn_calibration_histogram(conf, target, save_dir, method, nbins=20, epoch=9999):
    from sklearn.calibration import calibration_curve
    print('calibration sklearn')
    bin_size = [0.01, 0.05, 0.10]
    target_dict = {'IDH1':0, 'IDH2':1, 'IDH3':2, 'IDH4':3, 'IDH5':4}
    return_xaxis_dict = dict()
    return_yaxis_dict = dict()
    for step in bin_size :
        return_xaxis_dict[step] = dict()
        return_yaxis_dict[step] = dict()
        for idx, target_name in enumerate(target_dict.items()):
            nbins = int(1.0 /  step)
            fraction_of_positives, mean_predicted_value = calibration_curve(target[:,idx], conf[:,idx], n_bins=nbins)
            return_yaxis_dict[step][target_name] = fraction_of_positives
            return_xaxis_dict[step][target_name] = mean_predicted_value
            bins_ = np.insert(mean_predicted_value.copy(), len(mean_predicted_value), 1.0)

            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            xaxis_font_size = 20
            yaxis_font_size = 20
            tick_font_size = 20
            legend_font= 20
            fig = plt.figure(figsize=(8, 8)) 
            ax = plt.subplot()
            plt.tight_layout(rect=[0.18, 0.1, 0.96, 0.94])
            
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", color='b', alpha=1.0, label='{} Calibration'.format(method), )
            ax.plot(bins_, bins_, "k:" ,color='black', alpha=1.0, linestyle=':', label='Ideal Calibration')
            
            

            ax.set_ylim([0.0, 1.00001])
            ax.set_xlim([0.0, 1.00001])
            ax.set_yticks((0.0,0.25,0.5,0.75,1.0))
            ax.set_xticks((0.0,0.25,0.5,0.75,1.0))
            ax.set_xlabel('Confidence', fontsize=xaxis_font_size)
            ax.set_ylabel('Fraction of positives', fontsize=yaxis_font_size)
            ax.yaxis.set_label_coords(-0.21, 0.50)
            ax.xaxis.set_label_coords(0.50, -0.12)
            ax.grid(color='black', linestyle=':', alpha=0.8)
            # ax.set_title('calibration_{}'.format(key))
            ax.tick_params(direction='out', length=5, labelsize=tick_font_size, width=4, grid_alpha=0.5)
            ax.legend(loc='upper left', fontsize=legend_font)

            if not os.path.isdir('{}/calibration/bin_size_{:.2f}/'.format(save_dir, step)):
                os.makedirs('{}/calibration/bin_size_{:.2f}/'.format(save_dir, step))
            plt.savefig('{}/calibration/bin_size_{:.2f}/Line_graph_{}_calib_{}.jpg'.format(save_dir, step, target_name,epoch))
            plt.close()

    return return_yaxis_dict, return_xaxis_dict

def draw_confusion_matrix(total_conf, total_target, save_dir, threshold, epoch=9999):
    print('draw confusion matrix')
    save_dir += '/confusion_matrix'
    make_dir(save_dir)
    category = {'IDH1':0, 'IDH2':1, 'IDH3':2, 'IDH4':3, 'IDH5':4}
    for thres in threshold:
        for key, value in category.items():
            pred = (total_conf[:,value] > thres).long()
            target = total_target[:,value].long()

            # val_correct += (pred == target).sum().item()

            confusion_matrix_result, log = confusion_matrix(pred, target, 2)
            confusion_matrix_save_as_img(confusion_matrix_result.detach().cpu().numpy(),
                                                save_dir + '/{}'.format(thres),
                                                epoch, 'val', key)

# version3 용 eval.
def eval_rnn_classification_v3(loader, model, device, output_size, criterion, threshold=[0.5], log_dir=None, epoch=None, step=0, draw_confu=False):
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
        conf = F.sigmoid(total_output)
        sklearn_calibration_histogram(conf.detach().cpu().numpy(), total_target.detach().cpu().numpy(), save_dir=log_dir, method='RNN', epoch=epoch)
        result_dict = confidence_save_and_cal_auroc(conf, total_target, 'Validation', log_dir, epoch, step)
        if draw_confu :
            draw_confusion_matrix(conf, total_target, save_dir=log_dir, threshold=threshold,epoch=epoch)



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


def confidence_save_and_cal_auroc(conf, target, data_type, save_dir, epoch=9999, step=0):
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
        auroc, average_precision, [roc_tpr_array, roc_fpr_array, pr_precision_array, pr_recall_array] = roc_curve_plot_v2(load_dir=key_dir, category=key, conf=conf[:,value].detach().cpu().numpy(), target=target[:,value].detach().cpu().numpy(), epoch=epoch)
        # print("{} s |  {}  auroc : {}  AP : {} ".format(time.time() - start_time , key, auroc, average_precision))
        result_dict[key] = [auroc, average_precision]
    
    return result_dict