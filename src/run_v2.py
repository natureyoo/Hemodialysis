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

# hj
from torch.autograd import Variable
# from csv_parser import HemodialysisDataset

parser = argparse.ArgumentParser()
parser.add_argument('--save_result_root')
parser.add_argument('--bash_file')
parser.add_argument('--only_train')
parser.add_argument('--target_type')
parser.add_argument('--model_type')

parser.add_argument('--lr', type=float)
parser.add_argument('--lr_decay_rate', type=float)
# parser.add_argument('--lr_decay_epoch', default=[100,200,300,400])
parser.add_argument('--lr_decay_epoch', default=[15,40,70,100])
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--max_epoch', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--optim', required=False)
parser.add_argument('--loss', required=False)
parser.add_argument('--sampler', default=0, type=int)

parser.add_argument('--snapshot_epoch_freq', default=1, type=int)
parser.add_argument('--valid_iter_freq', default=500, type=int)
parser.add_argument('--description', default='')
parser.add_argument('--load_dir', default=None)
parser.add_argument('--merge_Flag', default=False)


args = parser.parse_args()


def mlp_classifications(args):
    input_size = 271
    hidden_size = args.hidden_size
    num_epochs = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    w_decay = args.weight_decay
    log_dir = args.save_result_root
    imbalanced = args.sampler
    target_type = args.target_type

    save_result = True
    sbp_target_idx = -2
    dbp_target_idx = -1
    # hj 6 -> 7
    sbp_num_class = 7
    dbp_num_class = 5
    output_size = sbp_num_class + dbp_num_class
    if args.merge_Flag :
        output_size = sbp_num_class * dbp_num_class
    stats = {'sbp_mean': 132.28392012972589, 'dbp_mean': 72.38757001151521, 'sbp_std': 26.86375195359048,
                 'dbp_std': 14.179178540137421}

    writer = SummaryWriter(log_dir + 'logs/')

    model = MLP_HJ(input_size, hidden_size, output_size).cuda()

    # if args.load_dir is not None:
    #     model = 

    train_data = torch.load('data/MLP/Train.pt')
    X = train_data[:, :-4]
    y = train_data[:, [sbp_target_idx, dbp_target_idx]] #Shape : (batch,2) --> (batch,1,1)
    val_data = torch.load('data/MLP/Validation.pt')
    X_val = val_data[:, :-4]
    y_val = val_data[:, [sbp_target_idx, dbp_target_idx]]

    # X, y = utils.data_preproc(X, y)

    # hj
    # remove label 3
    
    #########################################################################################
    # SBP ###################################################################################
    # label_train_3 = y[:,0] == 3.
    
    
    # # # sbp : True 100,000 --> idx 306,974 / True 150,000 --> idx 457,601
    # # # sbp : True 100,000 --> idx 185,788 / True 150,000 --> idx 276,149
    # # # count = 0
    # # # for i in range(len(label_dbp_2)):
    # # #     if label_dbp_2[i]:
    # # #         count += 1
    # # #     if count == 150000:
    # # #         print(i) 
    # # # print('total count: ', count)
    # # # exit()
    # label_train_3[457602:] = False
    # temp_x = X[label_train_3]
    # temp_y = y[label_train_3]

    # label_train_3 = y[:,0] != 3. 
    # X = X[label_train_3]
    # y = y[label_train_3]

    # X = np.concatenate((X, temp_x), axis=0)
    # y = np.concatenate((y, temp_y), axis=0)


    # # # 
    # # # label_val_3 = y_val[:,0] != 3.
    # # # X_val = X_val[label_val_3]
    # # # y_val = y_val[label_val_3]

    # # SBP - finish###########################################################################
    # #########################################################################################
    


    # #########################################################################################
    # # DBP ###################################################################################
    # label_train_dbp_2 = y[:,1] == 2.
    # label_train_dbp_2[276149:] = False
    # temp_x = X[label_train_dbp_2]
    # temp_y = y[label_train_dbp_2]

    # label_train_2 = y[:,1] != 2.
    # X = X[label_train_2]
    # y = y[label_train_2]

    # X = np.concatenate((X, temp_x), axis=0)
    # y = np.concatenate((y, temp_y), axis=0)

    # # label_val_2 = y_val[:,1] != 2.
    # # X_val = X_val[label_val_2]
    # # y_val = y_val[label_val_2]

    # # # hj #############################################
    # # np.random.seed(0)
    # # p = np.random.permutation(len(X))
    # # X = X[p]
    # # y = y[p]
    # # p = np.random.permutation(len(X_val))
    # # X_val = X_val[p]
    # # y_val = y_val[p]
    # # X = X[:10000]
    # # y = y[:10000]
    # # X_val = X_val[:10000]
    # # y_val = y_val[:10000]

    # # imbalanced_sampler = sampler.ImbalancedDatasetSampler(y_val, target_type)
    # imbalanced_sampler = sampler.ImbalancedDatasetSampler(y, target_type)
    # exit()




    train_dataset = loader.HD_Dataset((X, y))
    imbalanced_sampler = sampler.ImbalancedDatasetSampler(y, target_type)
    val_dataset = loader.HD_Dataset((X_val, y_val))

    if imbalanced == 1:
        print('imbala')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=imbalanced_sampler)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=512)

    # weight = torch.Tensor([])
    # idxs = [37917, 81139, 77997, 745146, 902692, \
    #     84212, 65156, 33607, 311857, 470365, \
    #         208057, 97855, 29595, 266478, 427933, \
    #             168896, 73089, 8887, 75705, 122490, \
    #                 621624, 285693, 34423, 105323, 190991, \
    #                     554787, 281717, 32115, 61398, 74532, \
    #                         1121246, 673970, 80410, 95600, 46106]
    # idxs = [0.5,0.4,0.3,0.2,0.1, \
    #     0.4,0.5,0.4,0.2,0.1, \
    #         0.1,0.4,0.5,0.4,0.1, \
    #             0.1,0.2,0.3,0.5,0.3, \
    #                 0.1,0.1,0.2,0.3,0.5,
    # ]
    temp = utils.sbp_dbp_target_converter(y.copy())
    uniques, counts = np.unique(temp, axis=0, return_counts=True)
    weight = torch.Tensor(counts)
    weight = weight.sum() / weight
    print(weight)
    sbp_criterion = nn.CrossEntropyLoss(weight=weight).cuda()
    # sbp_criterion = nn.CrossEntropyLoss().cuda()
    dbp_criterion = nn.CrossEntropyLoss().cuda()
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay, momentum=0.9)
        # , momentum=0.9
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    total_step = len(train_loader)
    best_acc = 0
    for epoch in range(num_epochs):
        if epoch in args.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate
        total = 0
        sbp_correct, dbp_correct = 0,0
        sbp_loss, dbp_loss = 0, 0
        sbp_loss_accum, dbp_loss_accum = 0, 0
        for i, (inputs, targets) in enumerate(train_loader):
            # if i == len(train_loader)-1:
            #     continue
            inputs = inputs.float().cuda().requires_grad_(True)
            targets = targets.long().cuda().requires_grad_(False)

            if args.merge_Flag :
                targets = utils.sbp_dbp_target_converter(targets)
            

            # Forward pass
            outputs = model(inputs)
            if not args.merge_Flag:
                sbp_outputs = outputs[:,:sbp_num_class]
                dbp_outputs = outputs[:,sbp_num_class:]

                _, sbp_pred = torch.max(sbp_outputs, 1)
                _, dbp_pred = torch.max(dbp_outputs, 1)

                sbp_loss = sbp_criterion(sbp_outputs, targets[:,0])
                dbp_loss = dbp_criterion(dbp_outputs, targets[:,1])

                total_loss = (sbp_loss + dbp_loss)

                # hj
                sbp_loss_accum = sbp_loss.detach().cpu() * (1./(i+1.)) + sbp_loss_accum * (i/(i+1.))
                dbp_loss_accum = dbp_loss.detach().cpu() * (1./(i+1.)) + dbp_loss_accum * (i/(i+1.))


                total += inputs.size(0)
                sbp_correct += (targets[:,0] == sbp_pred).sum().item()
                dbp_correct += (targets[:,1] == dbp_pred).sum().item()
            else : 
                outputs = outputs

                _, pred = torch.max(outputs, 1)
                total_loss = sbp_criterion(outputs, targets)
                sbp_loss_accum = total_loss.detach().cpu() * (1./(i+1.)) + sbp_loss_accum * (i/(i+1.))
                total += inputs.size(0)
                sbp_correct += (targets == pred).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            
            if (i + 1) % args.valid_iter_freq == 0:
                iteration = (i+1) + (total_step * epoch)
                print('Epoch [{}/{}]\tStep [{}/{}]\t|| SBP_loss: {:.4f}  DBP_loss: {:.4f}\tSBP_Acc: {:.4f}, DBP_Acc: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, sbp_loss_accum, dbp_loss_accum, sbp_correct / total, dbp_correct / total ))
                writer.add_scalars('Loss/Train', {'Total': total_loss,
                                                  'SBP': sbp_loss,
                                                  'DBP': dbp_loss}, iteration)
                writer.add_scalars('Acc/Train', {'SBP': sbp_correct / total,
                                                 'DBP': dbp_correct / total }, iteration)

                if not args.only_train :
                    continue
                with torch.no_grad():
                    model.eval()
                    val_total = 0
                    val_sbp_correct = 0
                    val_dbp_correct = 0
                    sbp_pred_tensor = torch.Tensor().long().cuda()
                    dbp_pred_tensor = torch.Tensor().long().cuda()
                    target_tensor = torch.Tensor().long().cuda()

                    for (inputs, targets) in val_loader:
                        inputs = inputs.float().cuda()
                        targets = targets.long().cuda()
                        outputs = model(inputs)
                        if args.merge_Flag:
                            targets = utils.sbp_dbp_target_converter(targets)
                            _, pred = torch.max(outputs, 1)
                            pred_s = utils.sbp_dbp_target_converter(pred, False)
                            sbp_pred = pred_s[:,0]
                            dbp_pred = pred_s[:,1]
                            targets = utils.sbp_dbp_target_converter(targets, False)

                        else:
                            sbp_outputs = outputs[:, :sbp_num_class]
                            dbp_outputs = outputs[:, sbp_num_class:]

                            _, sbp_pred = torch.max(sbp_outputs, 1)
                            _, dbp_pred = torch.max(dbp_outputs, 1)

                        val_total += inputs.size(0)
                        val_sbp_correct += (targets[:,0] == sbp_pred).sum().item()
                        val_dbp_correct += (targets[:,1] == dbp_pred).sum().item()
                        sbp_pred_tensor = torch.cat((sbp_pred_tensor, sbp_pred), dim=0)
                        dbp_pred_tensor = torch.cat((dbp_pred_tensor, dbp_pred), dim=0)
                        target_tensor = torch.cat((target_tensor, targets.long()), dim=0)

                        


                        
                    print("    || Acc. on Val || SBP: {:.3f}   DBP:{:.3f}".format(val_sbp_correct/ val_total, val_dbp_correct/ val_total))
                    writer.add_scalars('Acc/Val', {'SBP': val_sbp_correct / val_total,
                                                   'DBP': val_dbp_correct / val_total}, iteration)
                    

                    if (epoch+1) % args.snapshot_epoch_freq ==0 and (i+1) == total_step // args.valid_iter_freq * args.valid_iter_freq :
                        sbp_confus_matrix, sbp_log = utils.confusion_matrix(sbp_pred_tensor, target_tensor[:,0], sbp_num_class)
                        dbp_confus_matrix, dbp_log = utils.confusion_matrix(dbp_pred_tensor, target_tensor[:,1], dbp_num_class)
                        
                        utils.confusion_matrix_save_as_img(sbp_confus_matrix.detach().cpu().numpy(), args.save_result_root, epoch, 'sbp')
                        utils.confusion_matrix_save_as_img(dbp_confus_matrix.detach().cpu().numpy(), args.save_result_root, epoch, 'dbp')
                        for c in range(sbp_num_class):
                            per_class = {'Sensitivity': sbp_log[0]['class_{}'.format(c)], 'Specificity': sbp_log[1]['class_{}'.format(c)]}
                            writer.add_scalars('SBP_Metrics/class_{}'.format(c), per_class, iteration)
                        for c in range(dbp_num_class):
                            per_class = {'Sensitivity': dbp_log[0]['class_{}'.format(c)], 'Specificity': dbp_log[1]['class_{}'.format(c)]}
                            writer.add_scalars('DBP_Metrics/class_{}'.format(c), per_class, iteration)
                    curr_acc = (val_sbp_correct + val_dbp_correct) / 2  / val_total
                    if best_acc < curr_acc :
                        print("Saving best model with acc: {:.4f}...".format(curr_acc))
                        utils.save_snapshot(model, optimizer, args.save_result_root+'/best/', (epoch+1), iteration, (epoch+1))
                        best_acc = curr_acc
        if (epoch+1) % args.snapshot_epoch_freq == 0:
            utils.save_snapshot(model, optimizer, args.save_result_root, (epoch+1), 0, (epoch+1))


def main():
    args = parser.parse_args()
    print('args : ', args)
    args.save_result_root += args.model_type + '_' + args.target_type + '/'
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%b%d_%H%M%S_{}/".format(args.description))
    args.save_result_root += timestampStr
    print('\n|| save root : {}\n\n'.format(args.save_result_root))
    utils.copy_file(args.bash_file, args.save_result_root)  # .sh file 을 새 save_root에 복붙
    utils.copy_dir('./src', args.save_result_root+'src')    # ./src 에 code를 모아놨는데, src folder를 통째로 새 save_root에 복붙
    if args.model_type == 'mlp':
        mlp_classifications(args)

if __name__ == '__main__':
    main()
