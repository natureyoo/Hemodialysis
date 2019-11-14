import torch
from models import *
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import loader
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy.random as random
import sampler
import os

import sys
# from csv_parser import HemodialysisDataset

def parse_arg():
    parser = argparse.ArgumentParser(description='Prediction Blood Pressure during Hemodialysis using Deep Learning model')

    parser.add_argument('--save_result_root', type=str)
    parser.add_argument('--bash_file', type=str)
    parser.add_argument('--only_train')
    parser.add_argument('--target_type', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default 0.001)')
    parser.add_argument('--lr_decay_rate', type=float)
    parser.add_argument('--lr_decay_epoch', default=[50,100,150,200])
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--rnn_hidden_layers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optim', default='SGD')
    parser.add_argument('--loss', required=False)
    parser.add_argument('--sampler', default=False)
    parser.add_argument('--train_print_freq', default=5624, type=int)
    parser.add_argument('--dropout_rate', default=0.0, type=float)
    

    parser.add_argument('--snapshot_epoch_freq', default=1, type=int)
    parser.add_argument('--valid_iter_freq', default=500, type=int)
    parser.add_argument('--init_epoch', default=0, type=int)

    args = parser.parse_args()

    print('\n{}\n'.format(args))

    return args



def rnn_classification(args):
    input_size = 36
    # input_size = 143
    hidden_size = args.hidden_size
    num_layers = args.rnn_hidden_layers
    num_epochs = args.max_epoch
    output_size = 2
    num_class1 = 7
    num_class2 = 5
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    learning_rate = args.lr
    w_decay = args.weight_decay
    time = str(datetime.now())[:16].replace(' ', '_')
    task_type = args.target_type

    log_dir = '{}/bs{}_lr{}_wdecay{}'.format(args.save_result_root, batch_size, learning_rate, w_decay)
    utils.make_dir(log_dir)
    writer = SummaryWriter(log_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.model_type == 'rnn':
        model = RNN(input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate, task_type).to(device)
    elif args.model_type == 'rnn_v2':
        model = RNN_V2(input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate).to(device)
    else:
        print('model err')
        assert()
    
    #####################################################
    # load###############################################
    # print('load')
    # state = torch.load('./result/rnn_v2/Classification/Nov12_000400/bs16_lr0.01_wdecay5e-06/51epoch.model')
    # model.load_state_dict(state['model'])
    # load###############################################
    #####################################################

    # train_data = torch.load('./tensor_data/RNN/Train_5cut_0_40000.pt')
    train_data = torch.load('./tensor_data/RNN/Train.pt')
    # train_data = torch.load('./tensor_data/RNN/01/Test.pt')[:10000]
    
    len_tra = len(train_data)
    train_data = [train_data[i] for i in range(len(train_data)) if len(train_data[i])>6]
    for i in range(len(train_data)):
        train_data[i] = np.concatenate((train_data[i][:,:2], train_data[i][:,3:9],train_data[i][:,11:17], train_data[i][:,23:38], train_data[i][:,-12:-5], train_data[i][:,-4:]), axis=1)

    # weight1 = torch.Tensor([1.0500866,  0.8294839,  0.9128774,  0.2981994,  0.926195,   0.7277847, 1.0643833]).cuda()
    # weight2 = torch.Tensor([1.0443134,  0.8475585,  0.179056,   0.8153984,  0.9511724]).cuda()
    #############################################################################################################
    # weight1 = torch.Tensor([1.07397,  0.83084,  1.02054,  0.40,  1.00630,  0.72462, 1.11666]).cuda()
    # weight2 = torch.Tensor([1.09833,  0.85890,  0.20,  0.79806,  0.98818]).cuda()
    # tensor([10.7397,  8.3084,  9.2054,  2.9412,  9.0630,  7.2462, 11.1666],
    #    device='cuda:0') tensor([10.9833,  8.5890,  1.7667,  7.9806,  9.8818], device='cuda:0')

    ori_len = len(train_data)
    # train_data = [train_data[i] for i in range(300)]

    train_seq_len_list = [len(x) for x in train_data]

    print(len(train_seq_len_list))
    rm_list = list()
    for i in range(len(train_seq_len_list)):
        if train_seq_len_list[i] <= 6:
            print('1 - ', i, train_seq_len_list[i])
            rm_list.append(i)
    print(rm_list)
    count = 0
    for j in rm_list:
        print(j)
        del train_seq_len_list[j-count]
        np.delete(train_data, j-count) 
        count += 1
    
    for i in range(len(train_seq_len_list)):
        if train_seq_len_list[i] <= 6:
            print('2- ', i)
    print(len(train_data), len(train_seq_len_list))
    # exit()

    train_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in train_data])
    print("num train data : {} --> {}".format(ori_len, len(train_data)))
    del train_data
    print('del train data ok')
    train_data = loader.RNN_Dataset((train_padded, train_seq_len_list), type=task_type)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    # val_data = torch.load('./tensor_data/RNN/Validation.pt')
    val_data = torch.load('./tensor_data/RNN/Validation.pt')

    ####################################################
    # val_data = torch.load('./tensor_data/RNN/01/Validation.pt')[:10000]
    # val_data = [val_data[i] for i in range(len(val_data)) if len(val_data[i])>6]
    ########################################

    # len_val = len(val_data)
    # val_data = val_data[:int(len_val/10)]
    for i in range(len(val_data)):
        val_data[i] = np.concatenate((val_data[i][:,:2], val_data[i][:,3:9],val_data[i][:,11:17], val_data[i][:,23:38],val_data[i][:,-12:-5], val_data[i][:,-4:]), axis=1)
        # val_data[i] = np.concatenate((val_data[i][:,:38], val_data[i][:,-12:-11], val_data[i][:,-4:]), axis=1)

    val_seq_len_list = [len(x) for x in val_data]
    val_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in val_data])
    del val_data
    print('del val data ok')
    val_data = loader.RNN_Dataset((val_padded, val_seq_len_list), type=task_type)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)


    # weight1 = torch.Tensor([1.3,  1.2,  1.1,  0.5,  0.9,  0.7, 0.8]).cuda()
    # weight2 = torch.Tensor(          [1.0,  1.0,  0.2,  1.0, 1.0]          ).cuda()

    weight1 = torch.Tensor([1.3,  1.2,  1.1,  0.3,  0.9,  0.8, 0.7]).cuda()
    weight2 = torch.Tensor(          [1.0,  1.0,  0.15,  0.8, 0.9]          ).cuda()

    criterion1 = nn.CrossEntropyLoss(weight=weight1).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=weight2).to(device)

    criterion_sbp_KLD = nn.KLDivLoss().to(device)
    criterion_dbp_KLD = nn.KLDivLoss().to(device)
    # criterion1 = nn.CrossEntropyLoss().to(device)
    # criterion2 = nn.CrossEntropyLoss().to(device)
    
    
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay, momentum=0.9)
        # , momentum=0.9
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5,0.999), weight_decay=w_decay)
    elif args.optim == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        print('optim error')
        assert()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    best_loss = 100

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(args.init_epoch, num_epochs):
        if epoch in args.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate
                print('lr : {:.8f} --> {:.8f}'.format(param_group['lr']/args.lr_decay_rate, param_group['lr']))
        ##################################
        # eval ###########################
        model.eval()
        val_running_loss, val_size, pred1, pred2, flattened_target, sbp_accuracy, dbp_accuracy = utils.eval_rnn_classification(val_loader, model, device, output_size, criterion1, criterion2, num_class1, num_class2, log_dir=log_dir, epoch=epoch)
        sbp_confusion_matrix, sbp_log = utils.confusion_matrix(pred1, flattened_target[:, 0], num_class1)
        dbp_confusion_matrix, dbp_log = utils.confusion_matrix(pred2, flattened_target[:, 1], num_class2)
        utils.confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch , 0, 'val', 'sbp')
        utils.confusion_matrix_save_as_img(dbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch , 0, 'val', 'dbp')
        
        # eval ###########################
        ##################################
        exit()


        model.train()

        running_sbp_loss, running_dbp_loss = 0, 0
        total = 0
        train_total, train_correct_sbp, train_correct_dbp = 0,0,0 
        for batch_idx, (inputs, targets, seq_len) in enumerate(train_loader):
            inputs = inputs.permute(1,0,2).to(device)
            targets = targets.float().permute(1,0,2).to(device)
            seq_len = seq_len.to(device)

            output1, output2 = model(inputs, seq_len, device)


            flattened_output1 = torch.tensor([]).to(device)
            flattened_output2 = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                # flattened_output1 = torch.cat([flattened_output1, output1[seq-1, idx, :].reshape(-1, num_class1)], dim=0)
                # flattened_output2 = torch.cat([flattened_output2, output2[seq-1, idx, :].reshape(-1, num_class2)], dim=0)
                # flattened_target = torch.cat((flattened_target, targets[seq-1, idx, :].reshape(-1, output_size)), dim=0)

                flattened_output1 = torch.cat([flattened_output1, output1[:seq-6, idx, :].reshape(-1, num_class1)], dim=0)
                flattened_output2 = torch.cat([flattened_output2, output2[:seq-6, idx, :].reshape(-1, num_class2)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[6:seq, idx, :].reshape(-1, output_size)), dim=0)
            # loss1 = criterion1(flattened_output1, flattened_target[:, 0].long())
            # loss2 = criterion2(flattened_output2, flattened_target[:, 1].long())
            
            ##############################################################################################
            # soft labeling start ########################################################################
            # soft_target_sbp = utils.smooth_one_hot(flattened_target[:, 0])
            # soft_target_dbp = utils.smooth_one_hot(flattened_target[:, 1])
            soft_target_sbp = utils.smooth_one_hot(flattened_target[:, 0], smoothing=0.1, num_class=7).to(device)
            soft_target_dbp = utils.smooth_one_hot(flattened_target[:, 1], smoothing=0.1, num_class=5).to(device)
            loss1 = criterion_sbp_KLD(F.log_softmax(flattened_output1), soft_target_sbp)
            loss2 = criterion_dbp_KLD(F.log_softmax(flattened_output2), soft_target_dbp)

            # soft labeling finish #######################################################################
            ##############################################################################################
            if batch_idx < 10:
                utils.save_result_txt(torch.argmax(output1.permute(1,0,2), dim=2), targets[:,:, 0].permute(1,0), log_dir+'/txt/', epoch, 'Train_sbp', seq_lens=seq_len)
                utils.save_result_txt(torch.argmax(output2.permute(1,0,2), dim=2), targets[:,:, 1].permute(1,0), log_dir+'/txt/', epoch, 'Train_dbp', seq_lens=seq_len)
            
            ##########
            # hj
            # max_softmax_1 = F.softmax(flattened_output1)
            # max_softmax_2 = F.softmax(flattened_output2)
            # max1, pred1 = torch.max(max_softmax_1, 1)
            # max2, pred2 = torch.max(max_softmax_2, 1)
            # loss1 = MSELoss(max1, flattened_target[:, 0])
            # loss2 = MSELoss(max2, flattened_target[:, 1])
            # hj
            ############
            
            loss = loss1 + loss2
            # loss = loss1
            total += len(seq_len)

            # running_sbp_loss += loss1.item()
            # running_dbp_loss += loss2.item()

            # for param in model.parameters():
            #     print(param.name, param.grad)

            running_sbp_loss = loss1.item() * (1./(batch_idx+1.)) + running_sbp_loss * (batch_idx/(batch_idx+1.))
            running_dbp_loss = loss2.item() * (1./(batch_idx+1.)) + running_dbp_loss * (batch_idx/(batch_idx+1.))

            _, pred1 = torch.max(flattened_output1, 1)
            _, pred2 = torch.max(flattened_output2, 1)
            train_correct_sbp += (pred1 == flattened_target[:, 0].long()).sum().item()
            train_correct_dbp += (pred2 == flattened_target[:, 1].long()).sum().item()
            train_total += len(pred1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            # if (batch_idx + 1) % args.train_print_freq == 0:
            if epoch < 5:
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [{}/{}], Step [{}/{}], SBP loss: {:.4f}  DBP_loss: {:.4f}\t SBP acc.: {:.4f} DBP acc.: {:.4f}'
                    .format(epoch, num_epochs, batch_idx + 1, total_step, \
                            running_sbp_loss, running_dbp_loss, train_correct_sbp/train_total, train_correct_dbp/train_total))

                sys.stdout.flush()
            else:
                if batch_idx+1 == len(train_loader) :
                    print('Epoch [{}/{}], Step [{}/{}], SBP loss: {:.4f}  DBP_loss: {:.4f}\t SBP acc.: {:.4f} DBP acc.: {:.4f}'.format(
                        epoch, num_epochs, batch_idx + 1, total_step, \
                            running_sbp_loss, running_dbp_loss, train_correct_sbp/train_total, train_correct_dbp/train_total))
            
            if batch_idx+1 == len(train_loader) :
                # writer.add_scalar('Loss/Train', loss.item(), (i + 1) + total_step * epoch)
                # _, pred1 = torch.max(flattened_output1, 1)
                # _, pred2 = torch.max(flattened_output2, 1)
                # sbp_confusion_matrix, sbp_log = utils.confusion_matrix(pred1, flattened_target[:, 0].long(), num_class1)
                # dbp_confusion_matrix, dbp_log = utils.confusion_matrix(pred2, flattened_target[:, 1].long(), num_class2)
                # utils.confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch + 1, batch_idx + 1, 'train', 'sbp')
                # utils.confusion_matrix_save_as_img(dbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch + 1, batch_idx + 1, 'train', 'dbp')
                
                # if best_loss > val_running_loss :
                if epoch % 50 == 1 :
                    print("Saving model ...")
                    best_loss = val_running_loss
                    state = {'epoch': (epoch + 1), 'iteration': (batch_idx+1) + (total_step) * (epoch), 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, log_dir+'/{}epoch.model'.format(epoch))

                    # sbp_confusion_matrix, sbp_log = utils.confusion_matrix(pred1, flattened_target[:, 0], num_class1)
                    # dbp_confusion_matrix, dbp_log = utils.confusion_matrix(pred2, flattened_target[:, 1], num_class2)
                    # utils.confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch, batch_idx + 1, 'val', 'sbp')
                    # utils.confusion_matrix_save_as_img(dbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch, batch_idx + 1, 'val', 'dbp')
                    
                # writer.add_scalar('Loss/Val', val_running_loss/val_size, (batch_idx+1) +  total_step* epoch)
                # writer.add_scalar('SBP_Accuracy/Val', sbp_accuracy, (batch_idx + 1) + total_step * epoch)
                # writer.add_scalar('DBP_Accuracy/Val', dbp_accuracy, (batch_idx + 1) + total_step * epoch)

        # scheduler.step(((running_sbp_loss+running_dbp_loss)/2)/total)

    del train_loader, val_loader, train_padded, val_padded
    model.eval()

    print("\n\n\n ***Start testing***")
    test_data = torch.load('tensor_data/RNN/Test.pt')
    test_seq_len_list = [len(x) for x in test_data]
    test_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in test_data])
    test_data = loader.RNN_Dataset((test_padded, test_seq_len_list), type='Classification')
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    test_loss, test_size, _, _, _, sbp_accuracy, dbp_accuracy = utils.eval_rnn_classification(test_loader, model, device, output_size, criterion1, criterion2, num_class1, num_class2)
    print('test loss : {:.4f}'.format(test_loss))
    # writer.add_scalar('Loss/Test', test_loss/test_size, 1)


def main():
    args = parse_arg()
    args.save_result_root += '/' + args.model_type + '/' + args.target_type
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%b%d_%H%M%S/")
    args.save_result_root += '/' + timestampStr
    print('\n|| save root : {}\n\n'.format(args.save_result_root))
    utils.copy_file(args.bash_file, args.save_result_root)  # .sh file 을 새 save_root에 복붙
    utils.copy_dir('./src', args.save_result_root+'src')    # ./src 에 code를 모아놨는데, src folder를 통째로 새 save_root에 복붙

    rnn_classification(args)

if __name__ == '__main__':
    main()
