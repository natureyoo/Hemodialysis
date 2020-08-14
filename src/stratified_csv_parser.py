import pandas as pd
import os
import datetime as dt
import numpy as np
import json
import utils

import pickle

def HemodialysisDataset(path, files, save=True, EF_null_drop=False):
    print('save : {}'.format(save))
    print('EF_null_drop : {}'.format(EF_null_drop))
    hemodialysis_frame = pd.DataFrame()
    for file in files:
        tmp = pd.read_csv(os.path.join(path, file), header=0)
        hemodialysis_frame = pd.concat([hemodialysis_frame, tmp], ignore_index=True)
    
    print('total # of patient :', len(hemodialysis_frame.Pt_id.unique()))
    # hemodialysis_frame = hemodialysis_frame.loc[(hemodialysis_frame.VS_sbp > 0) & (hemodialysis_frame.VS_dbp > 0) & (hemodialysis_frame.HD_ctime > 0)]
    # hemodialysis_frame.drop(labels=['EF'], axis=1, inplace=True)

    np.random.seed(0)
    key = hemodialysis_frame['Pt_id'].unique()
    idx = list(np.random.permutation(range(len(key))))
    
    train_split, val_split = int(np.floor(0.6905 * len(key))), int(np.floor(0.8027 * len(key)))
    train_idx, val_idx, test_idx = idx[:train_split], idx[train_split:val_split], idx[val_split:]
    train_key, val_key, test_key = key[train_idx], key[val_idx], key[test_idx]

    np.random.seed(10)
    half_idx = list(np.random.permutation(range(len(val_key))))
    half_len = int(len(half_idx)/2)
    half_val_key = val_key[half_idx[:half_len]]
    cal_key = val_key[half_idx[half_len:]]
    
    
    
    data1 = hemodialysis_frame.loc[hemodialysis_frame['Pt_id'].isin(train_key)]
    data2 = hemodialysis_frame.loc[hemodialysis_frame['Pt_id'].isin(val_key)]
    data3 = hemodialysis_frame.loc[hemodialysis_frame['Pt_id'].isin(test_key)]

    if EF_null_drop :
        data1 = data1[~data1.EF.isnull()] # EF 변수 없는 환자 drop
        data2 = data2[~data2.EF.isnull()] # EF 변수 없는 환자 drop
        data3 = data3[~data3.EF.isnull()] # EF 변수 없는 환자 drop

    list_1 = data1.index.values
    list_2 = data2.index.values
    list_3 = data3.index.values
    
    if save :
        with open("../data/raw_data/train_idx.txt", "wb") as fp:   #Pickling
            pickle.dump(list_1, fp)
        with open("../data/raw_data/val_idx.txt", "wb") as fp:   #Pickling
            pickle.dump(list_2, fp)
        with open("../data/raw_data/test_idx.txt", "wb") as fp:   #Pickling
            pickle.dump(list_3, fp)

    
    # print(data1.head)
    # print(data2.head)
    # print(data3.head)
    print('patient tra/val/te:{} / {} / {}'.format(len(data1.Pt_id.unique()), len(data2.Pt_id.unique()), len(data3.Pt_id.unique())))
    if save :
        data1.to_csv('../data/raw_data/train.csv', index=False)
        data2.to_csv('../data/raw_data/val.csv', index=False)
        data3.to_csv('../data/raw_data/test.csv', index=False)


    data2_1 = hemodialysis_frame.loc[hemodialysis_frame['Pt_id'].isin(half_val_key)]
    data2_2 = hemodialysis_frame.loc[hemodialysis_frame['Pt_id'].isin(cal_key)]

    if EF_null_drop :
        data2_1 = data2_1[~data2_1.EF.isnull()] # EF 변수 없는 환자 drop
        data2_2 = data2_2[~data2_2.EF.isnull()] # EF 변수 없는 환자 drop

    list_2_1 = data2_1.index.values
    list_2_2 = data2_2.index.values
    if save:
        with open("../data/raw_data/half_val_idx.txt", "wb") as fp:   #Pickling
            pickle.dump(list_2_1, fp)
        with open("../data/raw_data/cal_idx.txt", "wb") as fp:   #Pickling
            pickle.dump(list_2_2, fp)

    print(data2_1.head)
    print(data2_2.head)
    
    print(len(data2_1.Pt_id.unique()), len(data2_2.Pt_id.unique()))
    if save:
        data2_1.to_csv('../data/raw_data/half_val.csv', index=False)
        data2_2.to_csv('../data/raw_data/cal.csv', index=False)

    print(len(list_2), len(list_2_1), len(list_2_2))


def check_HemodialysisDataset(path, files, save=True):
    unique_list = list()
    for file in files:
        tmp = pd.read_csv(os.path.join(path, file), header=0)
        print(len(tmp), len(tmp.Pt_id.unique()))
        unique_list.append(tmp.Pt_id.unique())
    
    num = 4
    for i in range(num) :
        for j in range(num) :
            count = 0
            if i != j :
                for k in range(len(unique_list[i])):
                    if unique_list[i][k] in unique_list[j]:
                        count += 1
            else:
                pass
            print(i, j, count)
        

def make_data(save, EF_null_drop):
    path ='../data/raw_data/'
    files = ['Hemodialysis1_0122.csv','Hemodialysis2_0122.csv']
    HemodialysisDataset(path, files, save=save, EF_null_drop=EF_null_drop )

def check_data():
    path ='../data/raw_data/'
    files = ['train.csv','train.csv', 'cal.csv', 'test.csv']
    check_HemodialysisDataset(path, files, save=True)

# tmp = pd.read_csv(os.path.join('../data/raw_data/Hemodialysis2_0122.csv'), header=0)
# tmp.to_csv('../data/raw_data/tmp2_0122.csv', index=False)
# print(tmp)
# exit()

# check_data()
# exit()



make_data(save=True, EF_null_drop=True)
