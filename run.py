from models import *
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import loader
import utils
import argparse
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import json
import sampler
import datetime
# from csv_parser import HemodialysisDataset

def mlp_regression(type):

    input_size = 269
    hidden_size = 64
    num_epochs = 3
    batch_size = 256
    learning_rate = 0.01
    w_decay = 0.0001
    num_class = 1
    time = str(datetime.datetime.now())[:16].replace(' ', '_')
    save_result = True
    target_type = type
    if target_type == 'sbp':
        target_idx = -3
    elif target_type =='dbp':
        target_idx = -4

    log_dir = 'result/mlp/regression/{}_{}_bs{}_lr{}_wdecay{}'.format(time, target_type, batch_size, learning_rate, w_decay)
    utils.make_dir(log_dir)
    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size, hidden_size,num_class).to(device)

    train_data = torch.load('tensor_data/MLP/Train.pt')
    X = train_data[:,:-4]
    y = train_data[:,target_idx]

    val_data = torch.load('tensor_data/MLP/Validation.pt')
    X_val = val_data[:,:-4]
    y_val = val_data[:,target_idx]

    train_dataset = loader.HD_Dataset((X,y))
    # imbalanced_sampler = sampler.ImbalancedDatasetSampler(y, target_type)
    val_dataset = loader.HD_Dataset((X_val, y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,verbose=True)
    best_loss = 100

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            # print("Input shape", inputs.shape)
            # print("Targets shape", targets.shape)
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs).float()
            targets = targets.float().view(-1, 1)
            # print("Outputs shape:", outputs.shape)

            loss = criterion(outputs, targets)
            total += inputs.size(0)
            running_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 500 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, running_loss/total))
                writer.add_scalar('Loss/Train', running_loss/total, (i+1) + (total_step) * (epoch))

                val_running_loss, val_size = utils.eval_regression(model, val_loader, device, log_dir)
                if best_loss > val_running_loss/ val_size:
                    print("Saving model ...".format(val_running_loss/val_size))
                    best_loss = val_running_loss / val_size
                    state = {'epoch': (epoch + 1), 'iteration': (i+1) + (total_step) * (epoch),
                             'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, log_dir+'/epoch{}_iter{}_loss{:.4f}.model'.format(epoch+1, (i+1) + (total_step) * (epoch), val_running_loss/val_size))
                print('\n')
                writer.add_scalar('Loss/Val', val_running_loss/val_size, (i+1) +  total_step* epoch)

    print("\n\n\n ***Start testing***")
    test_data = torch.load('tensor_data/MLP/Test.pt')
    X_test = test_data[:, :-4]
    y_test = test_data[:, target_idx]
    test_dataset = loader.HD_Dataset((X_test,y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_loss, test_size = utils.eval_regression(model,test_loader, device, log_dir, save_result, criterion)
    writer.add_scalar('Loss/Test', test_loss/test_size, 1 )

def mlp_cls(type):
    input_size = 269
    hidden_size = 64
    num_epochs = 3
    batch_size = 256
    learning_rate = 0.001
    w_decay = 0.0001
    time = str(datetime.datetime.now())[:16].replace(' ', '_')
    save_result = True
    target_type = type
    if target_type == 'sbp':
        target_idx = -2
        num_class = 5
    elif target_type =='dbp':
        target_idx = -1
        num_class = 4

    log_dir = 'result/mlp/cls/{}_{}_bs{}_lr{}_wdecay{}'.format(time, target_type, batch_size, learning_rate, w_decay)
    utils.make_dir(log_dir)
    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size, hidden_size, num_class).to(device)

    train_data = torch.load('tensor_data/MLP/Train.pt')
    X = train_data[:, :-4]
    y = train_data[:, target_idx]

    val_data = torch.load('tensor_data/MLP/Validation.pt')
    X_val = val_data[:, :-4]
    y_val = val_data[:, target_idx]

    train_dataset = loader.HD_Dataset((X, y))
    # imbalanced_sampler = sampler.ImbalancedDatasetSampler(y, target_type)
    val_dataset = loader.HD_Dataset((X_val, y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        correct= 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            # print("Input shape", inputs.shape)
            # print("Targets shape", targets.shape)
            inputs = inputs.float().to(device)
            targets = targets.long().to(device)

            # Forward pass
            outputs = model(inputs)
            targets = targets.long() - 1
            _, pred = torch.max(outputs, 1)

            loss = criterion(outputs, targets)
            total += inputs.size(0)
            correct += (targets == pred).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 500 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Acc: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, correct / total))
                # utils.compute_f1score(targets.data.to('cpu').numpy(), pred.data.to('cpu').numpy())

                with torch.no_grad():
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    pred_np = []
                    output_np = []

                    for (inputs, targets) in val_loader:
                        inputs = inputs.float().to(device)
                        targets = targets.long().to(device)

                        outputs = model(inputs)
                        _, pred = torch.max(outputs,1)

                        val_total += inputs.size(0)
                        val_correct += (targets == pred).sum().item()

                        pred_np.extend(pred.data.to('cpu').tolist())
                        output_np.extend(targets.data.to('cpu').tolist())

                    print("    Acc. on Validation: {:.3f}".format(correct/total))
                    # utils.compute_f1score(np.array(pred_np), np.array(output_np), True)


    # print("\n\n\n ***Start testing***")
    # test_data = torch.load('tensor_data/MLP/Test.pt')
    # X_test = test_data[:, :-4]
    # y_test = test_data[:, target_idx]
    # test_dataset = loader.HD_Dataset((X_test, y_test))
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # test_loss, test_size = utils.eval_regression(model, test_loader, device, log_dir, save_result, criterion)
    # writer.add_scalar('Loss/Test', test_loss / test_size, 1)
    #


def rnn_regression():
    input_size = 143
    hidden_size = 128
    num_layers = 2
    num_epochs = 1
    output_size = 2
    batch_size = 16
    dropout_rate = 0.2
    learning_rate = 0.005
    w_decay = 0.001
    time = str(datetime.datetime.now())[:16].replace(' ', '_')
    type = 'Regression'

    log_dir = 'result/rnn/{}/{}_bs{}_lr{}_wdecay{}'.format(type, time, batch_size, learning_rate, w_decay)
    utils.make_dir(log_dir)
    # writer = SummaryWriter(log_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate, type).to(device)

    train_data = torch.load('./tensor_data/RNN/Train.pt')
    train_seq_len_list = [len(x) for x in train_data]
    train_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in train_data])
    train_data = loader.RNN_Dataset((train_padded, train_seq_len_list), type=type)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    val_data = torch.load('./tensor_data/RNN/Validation.pt')
    val_seq_len_list = [len(x) for x in val_data]
    val_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in val_data])
    val_data = loader.RNN_Dataset((val_padded, val_seq_len_list), type=type)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    best_loss = 100

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0
        total = 0
        for i, (inputs, targets, seq_len) in enumerate(train_loader):
            inputs = inputs.permute(1,0,2).to(device)
            targets = targets.float().permute(1,0,2).to(device)
            seq_len = seq_len.to(device)

            outputs = model(inputs, seq_len, device)

            flattened_output = torch.tensor([]).to(device)
            flattened_target = torch.tensor([]).to(device)

            for idx, seq in enumerate(seq_len):
                flattened_output = torch.cat([flattened_output, outputs[:seq,idx,:].view(-1,output_size)], dim=0)
                flattened_target = torch.cat((flattened_target, targets[:seq,idx,:].view(-1,output_size)), dim=0)

            loss = criterion(flattened_target, flattened_output)
            total += len(seq_len)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, running_loss/total), end=' ')
                # writer.add_scalar('Loss/Train', loss.item(), (i + 1) + total_step * epoch)

                val_running_loss, val_size = utils.eval_rnn_regression(val_loader, model, device, output_size, criterion)
                if best_loss > val_running_loss:
                    print("Saving model ...")
                    best_loss = val_running_loss
                    state = {'epoch': (epoch + 1), 'iteration': (i+1) + (total_step) * (epoch), 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, log_dir+'/epoch{}_iter{}_loss{:.4f}.model'.format(epoch+1, (i+1) + (total_step) * (epoch), val_running_loss))
                # writer.add_scalar('Loss/Val', val_running_loss/val_size, (i+1) +  total_step* epoch)

    print("\n\n\n ***Start testing***")
    test_data = torch.load('tensor_data/RNN/Test.pt')
    test_seq_len_list = [len(x) for x in test_data]
    test_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in test_data])
    test_data = loader.RNN_Dataset((test_padded, test_seq_len_list), type='Regression')
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    test_loss, test_size = utils.eval_rnn_regression(test_loader, model, device, output_size, criterion)
    # writer.add_scalar('Loss/Test', test_loss/test_size, 1)


def rnn_classification():
    input_size = 143
    hidden_size = 128
    num_layers = 2
    num_epochs = 1
    output_size = 2
    num_class1 = 7
    num_class2 = 5
    batch_size = 16
    dropout_rate = 0.2
    learning_rate = 0.001
    w_decay = 0.001
    time = str(datetime.datetime.now())[:16].replace(' ', '_')
    type = 'Classification'

    log_dir = 'result/rnn/{}/{}_bs{}_lr{}_wdecay{}'.format(type, time, batch_size, learning_rate, w_decay)
    utils.make_dir(log_dir)
    # writer = SummaryWriter(log_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate, type).to(device)

    train_data = torch.load('./tensor_data/RNN/Train.pt')
    train_seq_len_list = [len(x) for x in train_data]
    train_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in train_data])
    train_data = loader.RNN_Dataset((train_padded, train_seq_len_list), type=type)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    val_data = torch.load('./tensor_data/RNN/Validation.pt')
    val_seq_len_list = [len(x) for x in val_data]
    val_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in val_data])
    val_data = loader.RNN_Dataset((val_padded, val_seq_len_list), type=type)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    best_loss = 100

    print("Starting training...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0
        total = 0
        for i, (inputs, targets, seq_len) in enumerate(train_loader):
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
            loss = loss1 + loss2
            total += len(seq_len)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, running_loss/total))
                # writer.add_scalar('Loss/Train', loss.item(), (i + 1) + total_step * epoch)

                val_running_loss, val_size = utils.eval_rnn_classification(val_loader, model, device, output_size, criterion1, criterion2, num_class1, num_class2)
                if best_loss > val_running_loss:
                    print("Saving model ...")
                    best_loss = val_running_loss
                    state = {'epoch': (epoch + 1), 'iteration': (i+1) + (total_step) * (epoch), 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, log_dir+'/epoch{}_iter{}_loss{:.4f}.model'.format(epoch+1, (i+1) + (total_step) * (epoch), val_running_loss))
                # writer.add_scalar('Loss/Val', val_running_loss/val_size, (i+1) +  total_step* epoch)

    print("\n\n\n ***Start testing***")
    test_data = torch.load('tensor_data/RNN/Test.pt')
    test_seq_len_list = [len(x) for x in test_data]
    test_padded = rnn_utils.pad_sequence([torch.tensor(x) for x in test_data])
    test_data = loader.RNN_Dataset((test_padded, test_seq_len_list), type='Regression')
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    test_loss, test_size = utils.eval_rnn_classification(test_loader, model, device, output_size, criterion, type)
    print('test loss : {:.4f}'.format(test_loss))
    # writer.add_scalar('Loss/Test', test_loss/test_size, 1)


mlp_cls('sbp')
# run_regression('dbp')
# rnn('sbp')