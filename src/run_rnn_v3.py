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
import sys

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

    # input_size = 36
    input_size = 143
    hidden_size = args.hidden_size
    num_layers = args.rnn_hidden_layers
    num_epochs = args.max_epoch
    output_size = 5
    num_class1 = 1
    num_class2 = 1
    num_class3 = 1
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
    elif args.model_type == 'rnn_v3':
        model = RNN_V3(input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate, num_class1, num_class2, num_class3).to(device)
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
    train_data = torch.load('./data/tensor_data/Interpolation_RNN_60min/New/Train1_60min.pt')
    train_data = np.concatenate([train_data, torch.load('./data/tensor_data/Interpolation_RNN_60min/New/Train2_60min.pt')], axis=0)
    train_data = np.concatenate([train_data, torch.load('./data/tensor_data/Interpolation_RNN_60min/New/Train3_60min.pt')], axis=0)
    train_data = np.concatenate([train_data, torch.load('./data/tensor_data/Interpolation_RNN_60min/New/Train4_60min.pt')], axis=0)

    # train_data = train_data[:int(len(train_data)*0.1)]              # using part of data

    # feature selection, manually.
    for i in range(len(train_data)):
        train_data[i] = train_data[i][:,1:] # remove masking
    ori_len = len(train_data)
    # Cut data by some rule
    #
    train_seq_len_list = [len(x) for x in train_data]
    print("num train data : {} --> {}".format(ori_len, len(train_data)))
    train_data = loader.RNN_Dataset((train_data, train_seq_len_list), type=task_type, ntime=60)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=loader.pad_collate)

    val_data = torch.load('./data/tensor_data/Interpolation_RNN_60min/New/Validation_60min.pt')
    val_data = val_data[:int(len(val_data) * 0.1)]
    val_seq_len_list = [len(x) for x in val_data]
    val_dataset = loader.RNN_Val_Dataset((val_data, val_seq_len_list), type=task_type, ntime=60)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False,
                            collate_fn=lambda batch: loader.pad_collate(batch, True))

    BCE_loss_with_logit = nn.BCEWithLogitsLoss().to(device)

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
        
        # ##################################
        # # eval ###########################
        threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
        model.eval()
        criterion = BCE_loss_with_logit
        utils.eval_rnn_classification_v3(val_loader, model, device, output_size, criterion, num_class1, num_class2, threshold, log_dir=log_dir, epoch=epoch)

        # 저장 : 매 epoch마다 하는데, 특정 epoch마다 하게 바꾸려면, epoch % args.print_freq == 0 등으로 추가
        state = {'epoch': (epoch + 1), 'iteration': 0, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, log_dir+'/{}epoch.model'.format(epoch))
        # # eval ###########################
        # ##################################


        model.train()

        running_loss, running_loss_sbp, running_loss_map, running_loss_under90, running_loss_sbp2, running_loss_map2 = 0,0,0,0,0,0
        
        total = 0
        train_total, train_correct_sbp,train_correct_map, train_correct_under_90, train_correct_sbp2, train_correct_map2  = 0,0,0,0,0,0
        for batch_idx, (inputs, (targets, targets_real), seq_len) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            seq_len = torch.LongTensor(seq_len).to(device)
            output = model(inputs, seq_len, device) # shape : (seq_len, batch size, 5)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                flattened_output = torch.cat([flattened_output, output[:seq, idx, :].reshape(-1, output_size)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq, idx, :].reshape(-1, output_size)), dim=0)
            
            loss_sbp = BCE_loss_with_logit(flattened_output[:,0], flattened_target[:,0])    # 이 loss는 알아서 input에 sigmoid를 씌워줌. 그래서 input : """logit""" / 단, target : 0 or 1
            loss_map = BCE_loss_with_logit(flattened_output[:,1], flattened_target[:,1])
            loss_under90 = BCE_loss_with_logit(flattened_output[:,2], flattened_target[:,2])
            loss_sbp2 = BCE_loss_with_logit(flattened_output[:,3], flattened_target[:,3])
            loss_map2 = BCE_loss_with_logit(flattened_output[:, 4], flattened_target[:, 4])
            
            # print('\n', F.sigmoid(flattened_target[0,0]).item(),  F.sigmoid(flattened_target[0,1]).item(),  F.sigmoid(flattened_target[0,2]).item())

            # if batch_idx < 10:    # 초기 10 iteration 의 결과를 txt 파일로 저장하겠다. # TODO : v2까지의 코드라서 v3에서 shape이 잘 안 맞을 가능성 있음
            #     utils.save_result_txt(torch.argmax(output1.permute(1,0,2), dim=2), targets[:,:, 0].permute(1,0), log_dir+'/txt/', epoch, 'Train_sbp', seq_lens=seq_len)
            #     utils.save_result_txt(torch.argmax(output2.permute(1,0,2), dim=2), targets[:,:, 1].permute(1,0), log_dir+'/txt/', epoch, 'Train_dbp', seq_lens=seq_len)
            
            loss = loss_sbp + loss_map + loss_under90 + loss_sbp2 + loss_map2

            running_loss_sbp = loss_sbp.item() * (1./(batch_idx+1.)) + running_loss_sbp * (batch_idx/(batch_idx+1.))
            running_loss_map = loss_map.item() * (1./(batch_idx+1.)) + running_loss_map * (batch_idx/(batch_idx+1.))
            running_loss_under90 = loss_under90.item() * (1./(batch_idx+1.)) + running_loss_under90 * (batch_idx/(batch_idx+1.))
            running_loss_sbp2 = loss_sbp2.item() * (1./(batch_idx+1.)) + running_loss_sbp2 * (batch_idx/(batch_idx+1.))
            running_loss_map2 = loss_map2.item() * (1./(batch_idx+1.)) + running_loss_map2 * (batch_idx/(batch_idx+1.))
            total += len(seq_len)

            # for param in model.parameters():
            #     print(param.name, param.grad) # gradient print

            running_loss = loss.item() * (1./(batch_idx+1.)) + running_loss * (batch_idx/(batch_idx+1.))

            pred0 = (F.sigmoid(flattened_output[:,0]) > 0.5).long()  # output : 1 or 0 --> 1: abnormal / 0: normal
            pred1 = (F.sigmoid(flattened_output[:,1]) > 0.5).long()
            pred2 = (F.sigmoid(flattened_output[:,2]) > 0.5).long()
            pred3 = (F.sigmoid(flattened_output[:,3]) > 0.5).long()
            pred4 = (F.sigmoid(flattened_output[:,4]) > 0.5).long()
            
            train_correct_sbp += (pred0 == flattened_target[:, 0].long()).sum().item() 
            train_correct_map += (pred1 == flattened_target[:, 1].long()).sum().item()
            train_correct_under_90 += (pred2 == flattened_target[:, 2].long()).sum().item()
            train_correct_sbp2 += (pred3 == flattened_target[:, 3].long()).sum().item()
            train_correct_map2 += (pred4 == flattened_target[:, 4].long()).sum().item()
            train_total += len(pred1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 3)
            optimizer.step()

            # if (batch_idx + 1) % args.train_print_freq == 0:
            if epoch < 5:       # 5 epoch 까지는 실시간으로 loss & acc를 보겠다.
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [{}/{}], Step [{}/{}], SBP l: {:.4f}  DBP_l: {:.4f} Under90 l:{:.4f}\t SBP acc.: {:.4f} MAP acc.: {:.4f} 90 acc.: {:.4f} SBP2 acc.: {:.4f}  MAP2 acc.: {:.4}'
                    .format(epoch, num_epochs, batch_idx + 1, total_step, \
                            running_loss_sbp, running_loss_map, running_loss_under90, train_correct_sbp/train_total, train_correct_map/train_total, train_correct_under_90/train_total, train_correct_sbp2/train_total, train_correct_map2/train_total))

                sys.stdout.flush()
            else:
                if batch_idx+1 == len(train_loader) :
                    print('| Epoch [{}/{}], Step [{}/{}], SBP l: {:.4f}  DBP_l: {:.4f} Under90 l:{:.4f}\t SBP acc.: {:.4f} MAP acc.: {:.4f} 90 acc.: {:.4f} SBP2 acc.: {:.4f}  MAP2 acc.: {:.4}'
                    .format(epoch, num_epochs, batch_idx + 1, total_step, \
                            running_loss_sbp, running_loss_map, running_loss_under90, train_correct_sbp/train_total, train_correct_map/train_total, train_correct_under_90/train_total, train_correct_sbp2/train_total, train_correct_map2/train_total))

    del train_loader, val_loader, train_padded, val_padded
    model.eval()

    ####################################################################3
    # TODO : test 
    print("\n\n\n ***Start testing***")
    test_data = torch.load('tensor_data/RNN/60min/Test.pt')
    test_seq_len_list = [len(x) for x in test_data]
    test_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in test_data])
    test_data = loader.RNN_Dataset((test_padded, test_seq_len_list), type='Classification', ntime=60)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    test_loss, test_size, _, _, _, sbp_accuracy, dbp_accuracy = utils.eval_rnn_classification(test_loader, model, device, output_size, criterion1, criterion2, num_class1, num_class2)
    print('test loss : {:.4f}'.format(test_loss))
    # writer.add_scalar('Loss/Test', test_loss/test_size, 1)
    ####################################################################3
    ####################################################################3


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
