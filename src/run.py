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
# from csv_parser import HemodialysisDataset

parser = argparse.ArgumentParser()
parser.add_argument('--save_result_root')
parser.add_argument('--bash_file')
parser.add_argument('--only_train')
parser.add_argument('--target_type')
parser.add_argument('--model_type')

parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--max_epoch', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--optim', required=False)
parser.add_argument('--loss', required=False)
parser.add_argument('--sampler', default=False)

parser.add_argument('--snapshot_epoch_freq', default=1, type=int)
parser.add_argument('--valid_iter_freq', default=500, type=int)

args = parser.parse_args()





def mlp_regression(args):

    input_size = 269
    hidden_size = args.hidden_size
    num_epochs = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    w_decay =args.weight_decay
    log_dir = args.save_result_root

    sbp_target_idx = -3
    dbp_target_idx = -4
    sbp_num_class = 1
    dbp_num_class = 1
    output_size = dbp_num_class + sbp_num_class
    stats = {'sbp_mean': 132.28392012972589, 'dbp_mean': 72.38757001151521, 'sbp_std': 26.86375195359048,
                 'dbp_std': 14.179178540137421}

    writer = SummaryWriter(log_dir + 'logs/')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size, hidden_size, output_size).to(device)

    train_data = torch.load('data/MLP/Train.pt')
    X = train_data[:,:-4]
    y = train_data[:,[sbp_target_idx, dbp_target_idx]]

    val_data = torch.load('data/MLP/Validation.pt')
    X_val = val_data[:,:-4]
    y_val = val_data[:,[sbp_target_idx, dbp_target_idx]]

    train_dataset = loader.HD_Dataset((X,y))
    # imbalanced_sampler = sampler.ImbalancedDatasetSampler(y, target_type)
    val_dataset = loader.HD_Dataset((X_val, y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,verbose=True)
    best_loss = 100

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        sbp_running_loss = 0
        dbp_running_loss = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            # print("Input shape", inputs.shape)
            # print("Targets shape", targets.shape)
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs).float()
            targets = targets.float().view(-1, 2)
            total += inputs.size(0)
            # print("Outputs shape:", outputs.shape)

            loss = criterion(outputs, targets)
            sbp_loss = loss[:,0]
            dbp_loss = loss[:,1]
            sbp_running_loss += sbp_loss.sum().item()
            dbp_running_loss += dbp_loss.sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            if (i + 1) % args.valid_iter_freq == 0:
                iteration = (i+1) + (total_step * epoch)
                print('Epoch [{}/{}], Step [{}/{}], SBP_Loss: {:.4f} DBP_Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, sbp_running_loss/total, dbp_running_loss/total))
                writer.add_scalar('SBP_Loss/Train', sbp_running_loss/total, (i+1) + (total_step) * (epoch))
                writer.add_scalar('DBP_Loss/Train', dbp_running_loss/total, (i+1) + (total_step) * (epoch))

                val_sbp_running_loss, val_dbp_running_loss, val_size = utils.eval_regression(model, val_loader, device, log_dir)
                val_running_loss = (val_sbp_running_loss + val_dbp_running_loss)
                if best_loss > val_running_loss/ val_size:
                    print("Saving best model ...".format(val_running_loss/val_size))
                    utils.save_snapshot(model, optimizer, args.save_result_root, (epoch+1), iteration, (epoch+1))
                    best_loss = val_running_loss / val_size
                print('\n')
                writer.add_scalar('Loss/Val', val_running_loss/val_size, (i+1 +  total_step* epoch))       # Plot outputs and targets

            if (epoch+1) % args.snapshot_epoch_freq == 0 and i == 0:
                sample_idx = random.choice(range(batch_size), size=50, replace=False)
                sample_sbp = (outputs[sample_idx,0] * stats['sbp_std'] + stats['sbp_mean'], targets[sample_idx,0] * stats['sbp_std'] + stats['sbp_mean'])
                sample_dbp = (outputs[sample_idx,1] * stats['dbp_std'] + stats['dbp_mean'], targets[sample_idx,1] * stats['dbp_std'] + stats['dbp_mean'])

                ax, plt = utils.save_plot(sample_sbp[0], sample_sbp[1], args.save_result_root, epoch + 1, 'train', 'SBP') # SBP
                plt.savefig(args.save_result_root+'/result/'+'sbp_{}epoch_{}.png'.format(epoch+1, "train"), dpi=300)
                ax, plt = utils.save_plot(sample_dbp[0], sample_dbp[1], args.save_result_root, epoch + 1, 'train', 'DBP') # DBP
                plt.savefig(args.save_result_root + '/result/' + 'dbp_{}epoch_{}.png'.format(epoch + 1, "train"), dpi=300)



    print("\n\n\n ***Start testing***")
    test_data = torch.load('data/MLP/Test.pt')
    X_test = test_data[:, :-4]
    y_test = test_data[:, [sbp_target_idx, dbp_target_idx]]
    test_dataset = loader.HD_Dataset((X_test,y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_sbp_loss, test_dbp_loss, test_size = utils.eval_regression(model,test_loader, device, log_dir, save_result, criterion)
    writer.add_scalar('SBP Loss/Test', test_sbp_loss/test_size, 1 )
    writer.add_scalar('DBP Loss/Test', test_dbp_loss/test_size, 1 )

def mlp_cls(args):
    input_size = 269
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
    sbp_num_class = 6
    dbp_num_class = 5
    output_size = dbp_num_class + sbp_num_class
    stats = {'sbp_mean': 132.28392012972589, 'dbp_mean': 72.38757001151521, 'sbp_std': 26.86375195359048,
                 'dbp_std': 14.179178540137421}

    writer = SummaryWriter(log_dir + 'logs/')

    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size, hidden_size, output_size).to(device)

    train_data = torch.load('data/MLP/Train.pt')
    X = train_data[:, :-4]
    y = train_data[:, [sbp_target_idx, dbp_target_idx]] #Shape : (batch,2) --> (batch,1,1)

    val_data = torch.load('data/MLP/Validation.pt')
    X_val = val_data[:, :-4]
    y_val = val_data[:, [sbp_target_idx, dbp_target_idx]]

    train_dataset = loader.HD_Dataset((X, y))
    imbalanced_sampler = sampler.ImbalancedDatasetSampler(y, target_type)
    val_dataset = loader.HD_Dataset((X_val, y_val))

    if imbalanced:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=imbalanced_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    sbp_criterion = nn.CrossEntropyLoss()
    dbp_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        sbp_correct = 0
        dbp_correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            # print("Input shape", inputs.shape)
            # print("Targets shape", targets.shape)
            inputs = inputs.float().to(device)
            targets = targets.long().to(device)

            # Forward pass
            outputs = model(inputs)
            sbp_outputs = outputs[:,:sbp_num_class]
            dbp_outputs = outputs[:,sbp_num_class:]

            _, sbp_pred = torch.max(sbp_outputs, 1)
            _, dbp_pred = torch.max(dbp_outputs, 1)

            sbp_loss = sbp_criterion(sbp_outputs, targets[:,0])
            dbp_loss = dbp_criterion(dbp_outputs, targets[:,1])
            total_loss = sbp_loss + dbp_loss

            total += inputs.size(0)
            sbp_correct += (targets[:,0] == sbp_pred).sum().item()
            dbp_correct += (targets[:,1] == dbp_pred).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i + 1) % args.valid_iter_freq == 0:
                iteration = (i+1) + (total_step * epoch)
                print('\n\nEpoch [{}/{}], Step [{}/{}], SBP_Acc: {:.4f}, DBP_Acc: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, sbp_correct / total, dbp_correct / total ))
                writer.add_scalars('Loss/Train', {'Total': total_loss,
                                                  'SBP': sbp_loss,
                                                  'DBP': dbp_loss}, iteration)
                writer.add_scalars('Acc/Train', {'SBP': sbp_correct / total,
                                                 'DBP': dbp_correct / total }, iteration)

                with torch.no_grad():
                    model.eval()
                    val_total = 0
                    val_sbp_correct = 0
                    val_dbp_correct = 0
                    sbp_pred_tensor = torch.Tensor().long().to(device)
                    dbp_pred_tensor = torch.Tensor().long().to(device)
                    target_tensor = torch.Tensor().long().to(device)

                    for (inputs, targets) in val_loader:
                        inputs = inputs.float().to(device)
                        targets = targets.long().to(device)

                        outputs = model(inputs)
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
                    print("\n    Acc. on Validation: SBP: {:.3f}   DBP:{:.3f}".format(val_sbp_correct/ val_total, val_dbp_correct/ val_total))
                    writer.add_scalars('Acc/Val', {'SBP': val_sbp_correct / val_total,
                                                   'DBP': val_dbp_correct / val_total}, iteration)

                    if (epoch+1) % args.snapshot_epoch_freq ==0 and (i+1) == total_step // args.valid_iter_freq * args.valid_iter_freq :
                        _, sbp_log = utils.confusion_matrix(sbp_pred_tensor, target_tensor[:,0], sbp_num_class)
                        _, dbp_log = utils.confusion_matrix(dbp_pred_tensor, target_tensor[:,1], dbp_num_class)
                        for c in range(sbp_num_class):
                            per_class = {'Sensitivity': sbp_log[0]['class_{}'.format(c)], 'Specificity': sbp_log[1]['class_{}'.format(c)]}
                            writer.add_scalars('SBP_Metrics/class_{}'.format(c), per_class, iteration)
                        for c in range(dbp_num_class):
                            per_class = {'Sensitivity': dbp_log[0]['class_{}'.format(c)], 'Specificity': dbp_log[1]['class_{}'.format(c)]}
                            writer.add_scalars('DBP_Metrics/class_{}'.format(c), per_class, iteration)



    # print("\n\n\n ***Start testing***")
    # test_data = torch.load('tensor_data/MLP/Test.pt')
    # X_test = test_data[:, :-4]
    # y_test = test_data[:, target_idx]
    # test_dataset = loader.HD_Dataset((X_test, y_test))
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # test_loss, test_size = utils.eval_regression(model, test_loader, device, log_dir, save_result, criterion)
    # writer.add_scalar('Loss/Test', test_loss / test_size, 1)
    #


if __name__ == '__main__':
    args.save_result_root += args.model_type + '_' + args.target_type + '/'
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%b%d_%H%M%S/")
    args.save_result_root += timestampStr
    print('\n|| save root : {}\n\n'.format(args.save_result_root))
    utils.copy_file(args.bash_file, args.save_result_root)  # .sh file 을 새 save_root에 복붙
    utils.copy_dir('./src', args.save_result_root+'src')    # ./src 에 code를 모아놨는데, src folder를 통째로 새 save_root에 복붙
    if args.target_type == 'regression':
        mlp_regression(args)
    elif args.target_type == 'cls':
        mlp_cls(args)

# run_regression('dbp')
# rnn('sbp')
