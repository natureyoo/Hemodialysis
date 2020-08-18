import torch
from models import *
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
import loader
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys,os

import random
import pandas as pd

def parse_arg():
    parser = argparse.ArgumentParser(description='Prediction Blood Pressure during Hemodialysis using Deep Learning model')

    parser.add_argument('--data_root', type=str, default='../data/raw_data/0813/pt_file_v1/wo_EF/')
    parser.add_argument('--save_result_root', type=str)
    parser.add_argument('--bash_file', type=str)
    parser.add_argument('--model_type', type=str, required=True)

    parser.add_argument('--input_fix_size', default=109, type=int)
    parser.add_argument('--input_seq_size', default=9, type=int)

    parser.add_argument('--optim', default='SGD')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default 0.001)')
    parser.add_argument('--lr_decay_rate', type=float)
    parser.add_argument('--dropout_rate', default=0.0, type=float)
    parser.add_argument('--cliping_grad', default=3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--rnn_hidden_layers', type=int, default=0)
    
    parser.add_argument('--init_epoch', default=0, type=int)
    parser.add_argument('--max_epoch', default=30, type=int)

    parser.add_argument('--load_path', default=None, type=str)

    parser.add_argument('--remove_version', default='no_remove')

    parser.add_argument('--gpu', default='1')

    parser.add_argument('--weight_loss_ratio', default=0.0, type=float, help='5.0, 3.0 etc....')
    parser.add_argument('--topk_loss_ratio', default=1.0, type=float, help='0.3, 0.1 etc....')
    parser.add_argument('--fc_initialize', default='gau', type=str, help='gau, xavier_unif')
    

    parser.add_argument('--result_csv_path', default='./result/result.csv')
    parser.add_argument('--result_csv_update', default=True)

    parser.add_argument('--draw_confusion_matrix', default=False)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    

    return args



def rnn_classification(args):
    input_fix_size = args.input_fix_size
    input_seq_size = args.input_seq_size
    hidden_size = args.hidden_size
    num_layers = args.rnn_hidden_layers
    num_epochs = args.max_epoch
    output_size = 5
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    learning_rate = args.lr
    w_decay = args.weight_decay

    result_dict = {'date':args.date[2:-1], 'stop_epo':0, 'IDH1':'0/0  (0)', 'IDH2':'0/0  (0)', 'IDH3':'0/0  (0)', 'IDH4':'0/0  (0)', 'IDH5':'0/0  (0)',  \
            'Optim':args.optim, 'lr':args.lr, 'wd':args.weight_decay, 'hidden_size':args.hidden_size, 'num_h_layer':args.rnn_hidden_layers, \
                'dropout':args.dropout_rate, 'batch size':args.batch_size, 'LOSS':'BCE', \
                'pos_weight':args.weight_loss_ratio, 'TopK':args.topk_loss_ratio, 'fc_initialize':args.fc_initialize}
    result_dict = pd.DataFrame.from_dict(result_dict, orient='index').T
    if os.path.exists(args.result_csv_path):
        result_df = pd.read_csv(args.result_csv_path, header=0)
        result_df = result_df.append(result_dict)
        result_df = result_df.reset_index(drop=True)
    else:
        result_df = result_dict
    

    log_dir = '{}/bs{}_lr{}_wdecay{}'.format(args.save_result_root, batch_size, learning_rate, w_decay)
    utils.make_dir(log_dir)
    utils.make_dir(log_dir+'/log/')
    writer = SummaryWriter(log_dir+'/log/')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RNN_V4(input_fix_size, input_seq_size, hidden_size, num_layers, batch_size, dropout_rate, initialize='gau').to(device)
        
    if args.load_path is not None:
        print('\n|| Load trained_model--> {}\n'.format(args.load_path))
        state = torch.load(args.load_path)
        model.load_state_dict(state['model'])
        # args.init_epoch = state['epoch']

    train_data = torch.load('{}/Train.pt'.format(args.data_root))

    # feature selection, manually.
    full_idx = [i for i in range(len(train_data[0][0]))]
    age = [1,]
    HD_info = list(range(2,7)) + list(range(8,15))
    HD_sum_IDH = list(range(21,26))
    VS = list(range(26,39))
    Lab=list(range(46,60))
    numeric = age + HD_info + HD_sum_IDH + VS + Lab + [i + len(full_idx) for i in range(-5,0)] # add HD_ntime_target
    onehot = [i for i in full_idx if i not in numeric]
    
    print('remove model: ', args.remove_version)
    print('one-hot:{}'.format(onehot))
    print('numeric:{}'.format(numeric))
    print('length : one {}      /     num {} (include 5-target)  '.format(len(onehot), len(numeric)))

    ori_len = len(train_data)
    train_data_ = []
    for i in range(len(train_data)):
        train_data_.append([train_data[i][0,onehot], train_data[i][:,numeric]])
    train_data = train_data_
    del train_data_

    train_seq_len_list = [len(x[1]) for x in train_data]
    print("num train data : {} --> {}".format(ori_len, len(train_data)))
    train_data = loader.RNN_Dataset((train_data, train_seq_len_list), ntime=60)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: loader.pad_collate(batch, True))

    val_data = torch.load('{}/Test.pt'.format(args.data_root))
    # val_data = val_data[:int(len(val_data) * 0.1)]
    val_data_ = []
    for i in range(len(val_data)):
        val_data_.append([val_data[i][0,onehot], val_data[i][:,numeric]])
    val_data = val_data_
    del val_data_

    val_seq_len_list = [len(x[1]) for x in val_data]
    val_dataset = loader.RNN_Dataset((val_data, val_seq_len_list), ntime=60)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False,
                            collate_fn=lambda batch: loader.pad_collate(batch, True))

    BCE_loss_with_logit = nn.BCEWithLogitsLoss().to(device)

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5,0.999), weight_decay=w_decay)
    elif args.optim == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    else:
        print('optim error')
        assert()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print('\n{}\n'.format(args))

    print("Starting training...")
    total_step = len(train_loader)
    max_result = {'IDH1':[0.0, 0.0, 0], 'IDH2':[0.0, 0.0, 0], 'IDH3':[0.0, 0.0, 0],'IDH4':[0.0, 0.0, 0],'IDH5':[0.0, 0.0, 0]}
    for epoch in range(args.init_epoch, num_epochs):
        if epoch > 0 :
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate
                print('lr : {:.8f} --> {:.8f}'.format(param_group['lr']/args.lr_decay_rate, param_group['lr']))
        
        model.eval()
        criterion = nn.BCEWithLogitsLoss().to(device)
        if epoch >= 0:
            result_dict = utils.eval_rnn_classification_v3(val_loader, model, device, output_size, criterion, [0.5], log_dir=log_dir, epoch=epoch, draw_confu=args.draw_confusion_matrix)
            for result_type_name, auroc_auprc_result_list in result_dict.items():
                if max_result[result_type_name][1] < round(auroc_auprc_result_list[1] * 100., 2) :
                    max_result[result_type_name][0] = round(auroc_auprc_result_list[0]* 100., 2) 
                    max_result[result_type_name][1] = round(auroc_auprc_result_list[1]* 100., 2) 
                    max_result[result_type_name][2] = epoch
                    result_df.iloc[len(result_df)-1, :].loc[result_type_name] = '{}/{}  ({})'.format(str(max_result[result_type_name][0]), str(max_result[result_type_name][1]), str(epoch))
            result_df.iloc[-1, :].loc['stop_epo'] = epoch
            if args.result_csv_update:
                result_df.to_csv(args.result_csv_path, index=False)


                
            print('max_dict : {}'.format(max_result))
            # state = {'epoch': epoch, 'iteration': 0, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            state = {'epoch': epoch, 'iteration': 0, 'model': model.state_dict()}
            utils.make_dir(log_dir+'/snapshot/')
            torch.save(state, log_dir+'/snapshot/{}epoch.model'.format(epoch))


        model.train()
        running_loss, running_loss_IDH1, running_loss_IDH2, running_loss_IDH3, running_loss_IDH4, running_loss_IDH5 = 0,0,0,0,0,0
        total = 0
        train_total, train_correct_IDH1, train_correct_IDH2, train_correct_IDH3, train_correct_IDH4, train_correct_IDH5  = 0,0,0,0,0,0
        for batch_idx, ((inputs_fix, inputs_seq), (targets), seq_len) in enumerate(train_loader):
            inputs_fix = inputs_fix.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = targets.float().to(device)
            seq_len = torch.LongTensor(seq_len).to(device)

            output = model(inputs_fix, inputs_seq, seq_len) # shape : (seq_len, batch size, 5)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                # rand_int = torch.randint(0, seq.item(), (1,)) 

                # flattened_output = torch.cat([flattened_output, output[rand_int:rand_int+1, idx, :].reshape(-1, output_size)], dim=0)
                # flattened_target = torch.cat((flattened_target, targets[rand_int:rand_int+1, idx, :].reshape(-1, output_size)), dim=0)
                flattened_output = torch.cat([flattened_output, output[:seq, idx, :].reshape(-1, output_size)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, output_size)), dim=0)
            # copy_idx = 2
            # flattened_target[:,0] = flattened_target[:,copy_idx]
            # flattened_target[:,1] = flattened_target[:,copy_idx]
            # flattened_target[:,2] = flattened_target[:,copy_idx]
            # flattened_target[:,3] = flattened_target[:,copy_idx]
            # flattened_target[:,4] = flattened_target[:,copy_idx]
            if args.weight_loss_ratio != 0.0:
                weight = torch.ones_like(flattened_target) / args.weight_loss_ratio
                crite_0 = nn.BCEWithLogitsLoss(pos_weight=weight[:,0]).to(device)
                crite_1 = nn.BCEWithLogitsLoss(pos_weight=weight[:,1]).to(device)
                crite_2 = nn.BCEWithLogitsLoss(pos_weight=weight[:,2]).to(device)
                crite_3 = nn.BCEWithLogitsLoss(pos_weight=weight[:,3]).to(device)
                crite_4 = nn.BCEWithLogitsLoss(pos_weight=weight[:,4]).to(device)
                loss_IDH1 = crite_0(flattened_output[:,0], flattened_target[:,0])
                loss_IDH2 = crite_1(flattened_output[:,1], flattened_target[:,1])
                loss_IDH3 = crite_2(flattened_output[:,2], flattened_target[:,2])
                loss_IDH4 = crite_3(flattened_output[:,3], flattened_target[:,3])
                loss_IDH5 = crite_4(flattened_output[:, 4], flattened_target[:, 4])
            if args.topk_loss_ratio != 1.0:
                BCE_loss_with_logit = nn.BCEWithLogitsLoss(reduction='none').to(device)
                loss_IDH1 = BCE_loss_with_logit(flattened_output[:,0], flattened_target[:,0])
                loss_IDH2 = BCE_loss_with_logit(flattened_output[:,1], flattened_target[:,1])
                loss_IDH3 = BCE_loss_with_logit(flattened_output[:,2], flattened_target[:,2])
                loss_IDH4 = BCE_loss_with_logit(flattened_output[:,3], flattened_target[:,3])
                loss_IDH5 = BCE_loss_with_logit(flattened_output[:, 4], flattened_target[:, 4])
                loss_IDH1, idx = torch.topk(loss_IDH1, int(args.topk_loss_ratio * loss_IDH1.size()[0]))
                loss_IDH2, idx = torch.topk(loss_IDH2, int(args.topk_loss_ratio * loss_IDH2.size()[0]))
                loss_IDH3, idx = torch.topk(loss_IDH3, int(args.topk_loss_ratio * loss_IDH3.size()[0]))
                loss_IDH4, idx = torch.topk(loss_IDH4, int(args.topk_loss_ratio * loss_IDH4.size()[0]))
                loss_IDH5, idx = torch.topk(loss_IDH5, int(args.topk_loss_ratio * loss_IDH5.size()[0]))
                loss_IDH1 = torch.mean(loss_IDH1)
                loss_IDH2 = torch.mean(loss_IDH2)
                loss_IDH3 = torch.mean(loss_IDH3)
                loss_IDH4 = torch.mean(loss_IDH4)
                loss_IDH5 = torch.mean(loss_IDH5)

            if (args.weight_loss_ratio == 0.0) and (args.topk_loss_ratio == 1.0):
                loss_IDH1 = BCE_loss_with_logit(flattened_output[:,0], flattened_target[:,0])
                loss_IDH2 = BCE_loss_with_logit(flattened_output[:,1], flattened_target[:,1])
                loss_IDH3 = BCE_loss_with_logit(flattened_output[:,2], flattened_target[:,2])
                loss_IDH4 = BCE_loss_with_logit(flattened_output[:,3], flattened_target[:,3])
                loss_IDH5 = BCE_loss_with_logit(flattened_output[:, 4], flattened_target[:, 4])

            loss = loss_IDH1 + loss_IDH2 + loss_IDH3 + loss_IDH4 + loss_IDH5

            running_loss_IDH1 = loss_IDH1.item() * (1./(batch_idx+1.)) + running_loss_IDH1 * (batch_idx/(batch_idx+1.))
            running_loss_IDH2 = loss_IDH2.item() * (1./(batch_idx+1.)) + running_loss_IDH2 * (batch_idx/(batch_idx+1.))
            running_loss_IDH3 = loss_IDH3.item() * (1./(batch_idx+1.)) + running_loss_IDH3 * (batch_idx/(batch_idx+1.))
            running_loss_IDH4 = loss_IDH4.item() * (1./(batch_idx+1.)) + running_loss_IDH4 * (batch_idx/(batch_idx+1.))
            running_loss_IDH5 = loss_IDH5.item() * (1./(batch_idx+1.)) + running_loss_IDH5 * (batch_idx/(batch_idx+1.))
            total += len(seq_len)

            running_loss = loss.item() * (1./(batch_idx+1.)) + running_loss * (batch_idx/(batch_idx+1.))

            pred0 = (F.sigmoid(flattened_output[:,0]) > 0.5).long()  # output : 1 or 0 --> 1: abnormal / 0: normal
            pred1 = (F.sigmoid(flattened_output[:,1]) > 0.5).long()
            pred2 = (F.sigmoid(flattened_output[:,2]) > 0.5).long()
            pred3 = (F.sigmoid(flattened_output[:,3]) > 0.5).long()
            pred4 = (F.sigmoid(flattened_output[:,4]) > 0.5).long()
            
            train_correct_IDH1 += (pred0 == flattened_target[:, 0].long()).sum().item() 
            train_correct_IDH2 += (pred1 == flattened_target[:, 1].long()).sum().item()
            train_correct_IDH3 += (pred2 == flattened_target[:, 2].long()).sum().item()
            train_correct_IDH4 += (pred3 == flattened_target[:, 3].long()).sum().item()
            train_correct_IDH5 += (pred4 == flattened_target[:, 4].long()).sum().item()
            train_total += len(pred1)

            optimizer.zero_grad()
            loss.backward()
            if args.cliping_grad is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.cliping_grad)
            optimizer.step()

            writer.add_scalar('Loss/Train', loss)

            if epoch < 5:
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [{}/{}], Step [{}/{}], LOSS:[IDH1:{:.4f}  IDH2:{:.4f} IDH3:{:.4f} IDH4:{:.4f}  IDH5: {:.4f} ] \t ACC:[IDH1:{:.4f}  IDH2:{:.4f} IDH3:{:.4f} IDH4:{:.4f}  IDH5: {:.4}]'
                    .format(epoch, num_epochs, batch_idx + 1, total_step, \
                            running_loss_IDH1, running_loss_IDH2, running_loss_IDH3, running_loss_IDH4, running_loss_IDH5,
                            train_correct_IDH1 / train_total,
                            train_correct_IDH2 / train_total, train_correct_IDH3 / train_total,
                            train_correct_IDH4 / train_total, train_correct_IDH5 / train_total))
                sys.stdout.flush()
            else:
                if batch_idx+1 == len(train_loader) :
                    print('| Epoch [{}/{}], Step [{}/{}], LOSS:[IDH1:{:.4f}  IDH2:{:.4f} IDH3:{:.4f} IDH4:{:.4f}  IDH5: {:.4f} ] \t ACC:[IDH1:{:.4f}  IDH2:{:.4f} IDH3:{:.4f} IDH4:{:.4f}  IDH5: {:.4}]'
                    .format(epoch, num_epochs, batch_idx + 1, total_step, \
                            running_loss_IDH1, running_loss_IDH2, running_loss_IDH3, running_loss_IDH4, running_loss_IDH5,
                            train_correct_IDH1 / train_total,
                            train_correct_IDH2 / train_total, train_correct_IDH3 / train_total,
                            train_correct_IDH4 / train_total, train_correct_IDH5 / train_total))
    model.eval()



def main():
    args = parse_arg()
    args.save_result_root += '/' + args.model_type + '/Classification'
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M%S/")
    args.date = timestampStr
    args.save_result_root += '/' + timestampStr
    print('\n|| save root : {}\n\n'.format(args.save_result_root))
    utils.copy_file(args.bash_file, args.save_result_root)
    utils.copy_dir('./src', args.save_result_root+'src')

    rnn_classification(args)

if __name__ == '__main__':
    main()