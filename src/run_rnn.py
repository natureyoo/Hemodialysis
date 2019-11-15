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
    parser.add_argument('--lr_decay_epoch', default=[30,60,100,150])
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

    args = parser.parse_args()

    print('\n{}\n'.format(args))

    return args



def rnn_classification(args):
    input_size = 143
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
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate, type).to(device)
    
    #####################################################
    # load###############################################
    # state = torch.load('/home/jayeon/Documents/code/Hemodialysis/result/rnn/Classification/Oct24_202805/bs16_lr0.0001_wdecay0.001/epoch13_iter147448_loss0.1959.model')
    # model.load_state_dict(state['model'])
    # load###############################################
    #####################################################

    train_data = torch.load('./tensor_data/RNN/Train_5cut_0_40000.pt')
    # tensor([10.2949,  7.9893,  8.7715,  2.6428,  9.1363,  8.2599, 18.2544],
    #    device='cuda:0') tensor([11.9553,  8.5961,  1.6335,  8.5410, 14.1324], device='cuda:0')
    weight1 = torch.Tensor([1.02949,  0.79893,  0.90715,  0.26428,  0.93363,  0.82599, 1.72544]).cuda()
    weight2 = torch.Tensor(          [1.19553,  0.90961,  0.16335,  0.90410, 1.31324]          ).cuda()


    #############################################################################################################
    # weight1 = torch.Tensor([1.07397,  0.83084,  1.02054,  0.40,  1.00630,  0.72462, 1.11666]).cuda()
    # weight2 = torch.Tensor([1.09833,  0.85890,  0.20,  0.79806,  0.98818]).cuda()
    # tensor([10.7397,  8.3084,  9.2054,  2.9412,  9.0630,  7.2462, 11.1666],
    #    device='cuda:0') tensor([10.9833,  8.5890,  1.7667,  7.9806,  9.8818], device='cuda:0')

    ori_len = len(train_data)
    # train_data = [train_data[i] for i in range(300)]

    train_seq_len_list = [len(x) for x in train_data]

    train_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in train_data])
    print("num train data : {} --> {}".format(ori_len, len(train_data)))
    del train_data
    print('del train data ok')
    train_data = loader.RNN_Dataset((train_padded, train_seq_len_list), type=task_type)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # val_data = torch.load('./tensor_data/RNN/Validation.pt')
    val_data = torch.load('./tensor_data/RNN/Valid_preproc.pt')

    val_seq_len_list = [len(x) for x in val_data]
    val_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in val_data])
    del val_data
    print('del val data ok')
    val_data = loader.RNN_Dataset((val_padded, val_seq_len_list), type=task_type)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    criterion1 = nn.CrossEntropyLoss(weight=weight1).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=weight2).to(device)
    
    MSELoss = nn.MSELoss()
    
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay, momentum=0.9)
        # , momentum=0.9
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    best_loss = 100

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        if epoch in args.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate
                print('lr : {:.8f} --> {:.8f}'.format(param_group['lr']/args.lr_decay_rate, param_group['lr']))
        
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
                flattened_output1 = torch.cat([flattened_output1, output1[:seq, idx, :].reshape(-1, num_class1)], dim=0)
                flattened_output2 = torch.cat([flattened_output2, output2[:seq, idx, :].reshape(-1, num_class2)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, output_size)), dim=0)
            loss1 = criterion1(flattened_output1, flattened_target[:, 0].long())
            loss2 = criterion2(flattened_output2, flattened_target[:, 1].long())
            
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
            total += len(seq_len)
            
            running_sbp_loss = loss1.item() * (1./(batch_idx+1.)) + running_sbp_loss * (batch_idx/(batch_idx+1.))
            running_dbp_loss = loss2.item() * (1./(batch_idx+1.)) + running_dbp_loss * (batch_idx/(batch_idx+1.))

            _, pred1 = torch.max(flattened_output1, 1)
            _, pred2 = torch.max(flattened_output2, 1)
            train_correct_sbp += (pred1 == flattened_target[:, 0].long()).sum().item()
            train_correct_dbp += (pred2 == flattened_target[:, 1].long()).sum().item()
            train_total += len(pred1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (batch_idx + 1) % args.train_print_freq == 0:
            if batch_idx+1 == len(train_loader) :
                print('Epoch [{}/{}], Step [{}/{}], SBP loss: {:.4f}  DBP_loss: {:.4f}\t SBP acc.: {:.4f} DBP acc.: {:.4f}'.format(
                    epoch, num_epochs, batch_idx + 1, total_step, \
                        running_sbp_loss/total, running_dbp_loss/total, train_correct_sbp/train_total, train_correct_dbp/train_total
                        )
                            )
                # writer.add_scalar('Loss/Train', loss.item(), (i + 1) + total_step * epoch)
                # _, pred1 = torch.max(flattened_output1, 1)
                # _, pred2 = torch.max(flattened_output2, 1)
                # sbp_confusion_matrix, sbp_log = utils.confusion_matrix(pred1, flattened_target[:, 0].long(), num_class1)
                # dbp_confusion_matrix, dbp_log = utils.confusion_matrix(pred2, flattened_target[:, 1].long(), num_class2)
                # utils.confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch + 1, batch_idx + 1, 'train', 'sbp')
                # utils.confusion_matrix_save_as_img(dbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch + 1, batch_idx + 1, 'train', 'dbp')

                model.eval()

                val_running_loss, val_size, pred1, pred2, flattened_target, sbp_accuracy, dbp_accuracy = utils.eval_rnn_classification(val_loader, model, device, output_size, criterion1, criterion2, num_class1, num_class2)

                sbp_confusion_matrix, sbp_log = utils.confusion_matrix(pred1, flattened_target[:, 0], num_class1)
                dbp_confusion_matrix, dbp_log = utils.confusion_matrix(pred2, flattened_target[:, 1], num_class2)
                utils.confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch , batch_idx + 1, 'val', 'sbp')
                utils.confusion_matrix_save_as_img(dbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch , batch_idx + 1, 'val', 'dbp')

                
                # if best_loss > val_running_loss :
                if epoch % 50 == 1 :
                    print("Saving model ...")
                    best_loss = val_running_loss
                    state = {'epoch': (epoch + 1), 'iteration': (batch_idx+1) + (total_step) * (epoch), 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, log_dir+'/{}epoch.model'.format(epoch))

                    sbp_confusion_matrix, sbp_log = utils.confusion_matrix(pred1, flattened_target[:, 0], num_class1)
                    dbp_confusion_matrix, dbp_log = utils.confusion_matrix(pred2, flattened_target[:, 1], num_class2)
                    utils.confusion_matrix_save_as_img(sbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch, batch_idx + 1, 'val', 'sbp')
                    utils.confusion_matrix_save_as_img(dbp_confusion_matrix.detach().cpu().numpy(), log_dir, epoch, batch_idx + 1, 'val', 'dbp')
                    
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