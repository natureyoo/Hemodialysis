from torch.utils.data import Dataset
import pandas as pd
import os
import datetime as dt
import numpy as np
import torch

class HemodialysisDataset():
    """Hemodialysis dataset from SNU"""

    def __init__(self, root_dir, csv_files, model_type, save=True):
        self.hemodialysis_frame = pd.DataFrame()
        self.files = csv_files
        self.root_dir = root_dir
        # self.training_type = training_type
        self.model_type = model_type
        self.total_seq = []
        self.columns = []
        self.mean_for_normalize = {}
        self.std_for_normalize = {}
        # self.init_value = pd.DataFrame()
        self.sbp_diff = -20
        self.dbp_diff = -10
        self.seed= 212
        self._init_dataset(save)

    def __len__(self):
        return len(self.hemodialysis_frame)

    def __getitem__(self, idx):
        return self.hemodialysis_frame[idx]

    def _init_dataset(self, save):
        for f in self.files:
            tmp = pd.read_csv(os.path.join(self.root_dir, f), header=0)
            self.hemodialysis_frame = pd.concat([self.hemodialysis_frame, tmp])
        self.refine_dataset()
        self.normalize()
        # self.hemodialysis_frame = self.hemodialysis_frame.loc[self.hemodialysis_frame.ID_class == self.training_type]
        if self.model_type == 'MLP':
            self.concat_history()
        else:
            self.make_sequence()
            self.add_pre_hemodialysis()
            self.order_target_column()
        self.split_train_test()
        self.columns = list(filter(lambda x: x not in ['ID_class', 'ID_hd'], self.hemodialysis_frame.columns))
        if save:
            print("Saving...")
            for type in ['Train', 'Validation', 'Test']:
                df = self.hemodialysis_frame.loc[self.hemodialysis_frame.ID_class == type].copy()
                df = df.drop('ID_class', axis=1)
                if self.model_type == 'MLP':
                    df = df.drop('ID_hd', axis=1)
                    df = np.array(df).astype('float')
                else:
                    df = self.convert_to_sequence(df)
                torch.save(df, '{}_{}.pt'.format(self.model_type, type))
        if not save:
            self.hemodialysis_frame.drop('ID_class', axis=1, inplace=True)
            if self.model_type == 'MLP':
                np.array(self.hemodialysis_frame).astype('float')
            else:
                self.hemodialysis_frame = self.convert_to_sequence(self.hemodialysis_frame)


    def split_train_test(self):
        print("Splitting...")
        np.random.seed(self.seed)
        key = self.hemodialysis_frame['ID_hd'].unique()
        idx = np.random.permutation(range(len(key)))
        train_split, val_split = int(np.floor(0.7 * len(key))), int(np.floor(0.8 * len(key)))
        train_idx, val_idx, test_idx = idx[:train_split], idx[train_split:val_split], idx[val_split:]
        training_type = pd.DataFrame(data=key, columns=['ID_hd'])
        training_type['ID_class'] = 0; training_type.loc[train_idx, 'ID_class'] = 'Train'; training_type.loc[val_idx, 'ID_class'] = 'Validation'; training_type.loc[test_idx, 'ID_class'] = 'Test'
        self.hemodialysis_frame = training_type.merge(self.hemodialysis_frame, how='inner', left_on=["ID_hd"], right_on=["ID_hd"])

    def refine_dataset(self):
        self.hemodialysis_frame['Pt_id'] = [x.replace('*', '') for x in self.hemodialysis_frame['Pt_id']]
        self.hemodialysis_frame['Pt_id'] = self.hemodialysis_frame['Pt_id'].astype(str).astype(int)
        self.hemodialysis_frame['ID_hd'] = [x.replace('*', '') for x in self.hemodialysis_frame['ID_hd']]
        self.hemodialysis_frame['ID_hd'] = self.hemodialysis_frame['ID_hd'].astype(str).astype(int)
        categorical = ['HD_type','HD_acces','HD_prim','HD_dialysate','HD_dialyzer']
        drop_columns = ['VS_rr', 'Time']
        self.hemodialysis_frame.drop(labels=drop_columns, inplace=True, axis=1)
        timestamp = pd.to_datetime(self.hemodialysis_frame['HD_duration']).dt
        self.hemodialysis_frame['HD_duration'] = timestamp.hour * 60 + timestamp.minute
        self.hemodialysis_frame.fillna(method='ffill', inplace=True)
        self.hemodialysis_frame['Pt_sex'] = self.hemodialysis_frame['Pt_sex'].replace({'M':0, 'F':1})
        self.hemodialysis_frame = pd.get_dummies(self.hemodialysis_frame, columns=categorical, prefix=categorical)
        self.hemodialysis_frame = self.hemodialysis_frame.loc[(self.hemodialysis_frame.VS_sbp > 0) & (self.hemodialysis_frame.VS_dbp > 0)]

    def concat_history(self):
        print("Concating...")
        self.hemodialysis_frame.drop(labels=['Pt_id'], axis=1, inplace=True)
        self.hemodialysis_frame['time'] = [dt.datetime.strptime(x[10:], '%Y%m%d*%H%M') for x in self.hemodialysis_frame['ID_timeline']]
        self.hemodialysis_frame['prev_time'] = self.hemodialysis_frame.apply(lambda x: x.time - dt.timedelta(minutes=x.HD_ntime_raw), axis=1)
        prev_data = self.hemodialysis_frame.copy()
        target_data = self.hemodialysis_frame.loc[self.hemodialysis_frame.HD_ctime_raw > 0][['ID_hd','prev_time', 'HD_ntime', 'VS_sbp', 'VS_dbp']]
        init_data = self.hemodialysis_frame.loc[self.hemodialysis_frame.HD_ctime_raw == 0][['ID_hd', 'VS_sbp', 'VS_dbp']]
        self.hemodialysis_frame = self.hemodialysis_frame.merge(prev_data, how='inner', left_on=['ID_hd', 'prev_time'], right_on=['ID_hd', 'time'], suffixes=('', '_prev'))
        self.hemodialysis_frame = self.hemodialysis_frame.merge(target_data, how='inner', left_on=['ID_hd', 'time'], right_on=['ID_hd', 'prev_time'], suffixes=('', '_target'))
        self.hemodialysis_frame = init_data.merge(self.hemodialysis_frame, how='inner', on=['ID_hd'], suffixes=('_init', ''))
        drop_columns_for_learning = ['ID_timeline', 'HD_ctime_raw', 'HD_ntime_raw', 'time', 'prev_time', 'Pt_sex_prev', 'Pt_age_prev', 'ID_timeline_prev', 'time_prev', 'prev_time_prev', 'prev_time_target']
        self.hemodialysis_frame.drop(labels=drop_columns_for_learning, axis=1, inplace=True)
        self.add_target_class()
        # self.init_value = self.hemodialysis_frame[['VS_sbp_init', 'VS_dbp_init']]

    def make_sequence(self):
        self.hemodialysis_frame['time'] = [dt.datetime.strptime(x[10:], '%Y%m%d*%H%M') for x in self.hemodialysis_frame['ID_timeline']]
        self.hemodialysis_frame['prev_time'] = self.hemodialysis_frame.apply(lambda x: x.time - dt.timedelta(minutes=x.HD_ntime_raw), axis=1)
        target_data = self.hemodialysis_frame.loc[self.hemodialysis_frame.HD_ctime_raw > 0][['ID_hd', 'prev_time', 'HD_ntime', 'VS_sbp', 'VS_dbp']]
        init_data = self.hemodialysis_frame.loc[self.hemodialysis_frame.HD_ctime_raw == 0][['ID_hd', 'VS_sbp', 'VS_dbp']]
        self.hemodialysis_frame = self.hemodialysis_frame.merge(target_data, how='inner', left_on=['ID_hd', 'time'], right_on=['ID_hd', 'prev_time'], suffixes=('', '_target'))
        self.hemodialysis_frame = init_data.merge(self.hemodialysis_frame, how='inner', on=['ID_hd'], suffixes=('_init', ''))

        drop_columns = ['ID_timeline', 'HD_ntime_raw', 'time', 'prev_time', 'prev_time_target']
        self.hemodialysis_frame.drop(labels=drop_columns, axis=1, inplace=True)
        self.add_target_class()

    def normalize(self):
        print('Normalizing...')
        self.hemodialysis_frame['HD_ctime_raw'] = self.hemodialysis_frame['HD_ctime']
        self.hemodialysis_frame['HD_ntime_raw'] = self.hemodialysis_frame['HD_ntime']
        numerical_col = ['Pt_age', 'HD_ntime', 'HD_ctime', 'HD_prewt', 'HD_uf', 'VS_sbp', 'VS_dbp', 'VS_hr', 'VS_bt', 'VS_bfr', 'VS_uft', 'Lab_wbc', 'Lab_hb', 'Lab_plt', 'Lab_chol', 'Lab_alb', 'Lab_glu', 'Lab_ca', 'Lab_phos', 'Lab_ua', 'Lab_bun', 'Lab_scr', 'Lab_na', 'Lab_k', 'Lab_cl', 'Lab_co2']
        for col in numerical_col:
            self.mean_for_normalize[col] = self.hemodialysis_frame[col].mean()
            self.std_for_normalize[col] = self.hemodialysis_frame[col].std()
            if self.std_for_normalize[col] > 0:
                self.hemodialysis_frame[col] = (self.hemodialysis_frame[col] - self.mean_for_normalize[col]) / self.std_for_normalize[col]
            else:
                self.hemodialysis_frame[col] = 0

    def add_target_class(self):
        def eval_target(diff, type):
            if type == 'sbp':
                if diff < -20 :
                    return 0
                if diff < -10 :
                    return 1
                if diff < -5 :
                    return 2
                if diff < 5 :
                    return 3
                if diff < 15:
                    return 4
                else:
                    return 5

            if type == 'dbp':
                if diff < -10:
                    return 0
                if diff < -5:
                    return 1
                if diff < 5:
                    return 2
                if diff < 10:
                    return 3
                else:
                    return 4

        self.hemodialysis_frame['VS_sbp_target_class'] = ((self.hemodialysis_frame['VS_sbp_target'] - self.hemodialysis_frame['VS_sbp']) * self.std_for_normalize['VS_sbp']).apply(lambda x: eval_target(x,'sbp'))
        self.hemodialysis_frame['VS_dbp_target_class'] = ((self.hemodialysis_frame['VS_dbp_target'] - self.hemodialysis_frame['VS_dbp']) * self.std_for_normalize['VS_dbp']).apply(lambda x: eval_target(x,'dbp'))

    def add_pre_hemodialysis(self):
        def merge_pre(frame):
            min_bp = frame.groupby(['Pt_id', 'ID_hd'])['VS_sbp', 'VS_dbp'].min().reset_index()
            max_bp = frame.groupby(['Pt_id', 'ID_hd'])['VS_sbp', 'VS_dbp'].max().reset_index()
            init_bp = frame.loc[frame.HD_ctime_raw == 0][['Pt_id', 'ID_hd', 'VS_sbp', 'VS_dbp']]
            init_bp.columns = ['Pt_id', 'ID_hd', 'VS_sbp_init_in_pre_hd', 'VS_dbp_init_in_pre_hd']
            pre_hd = min_bp.merge(max_bp, how='inner', on=['Pt_id', 'ID_hd'], suffixes=('_min_in_pre_hd', '_max_in_pre_hd'))
            pre_hd = init_bp.merge(pre_hd, how='inner', on=['Pt_id', 'ID_hd'])
            pre_hd['rank'] = pre_hd.sort_values(['Pt_id', 'ID_hd'], ascending=[True, True]).groupby(['Pt_id']).cumcount() + 1
            del min_bp, max_bp, init_bp
            frame = frame.merge(pre_hd[['Pt_id', 'ID_hd', 'rank']], on=['Pt_id', 'ID_hd'], how='inner')
            frame['rank'] -= 1
            frame = frame.merge(pre_hd, on=['Pt_id', 'rank'], how='left', suffixes=('', '_pre'))
            frame['pre_hd'] = [0 if np.isnan(x) else 1 for x in frame['VS_sbp_min_in_pre_hd']]
            frame.drop(labels=['Pt_id', 'rank', 'ID_hd_pre', 'HD_ctime_raw'], axis=1, inplace=True)
            frame.fillna(0.0, inplace=True)
            return frame
        self.hemodialysis_frame = merge_pre(self.hemodialysis_frame)

    def order_target_column(self):
        target_columns = ['VS_sbp_target', 'VS_dbp_target', 'VS_sbp_target_class', 'VS_dbp_target_class']
        input_columns = self.hemodialysis_frame.columns[
            [False if x in target_columns else True for x in self.hemodialysis_frame.columns]].to_list()
        self.hemodialysis_frame = self.hemodialysis_frame[input_columns + target_columns]

    def convert_to_sequence(self, df):
        grouped = df.sort_values(['ID_hd', 'HD_ctime'], ascending=[True,True]).groupby('ID_hd')
        unique = df['ID_hd'].unique()
        for id_ in unique:
            seq = grouped.get_group(id_)  # dataframe type
            seq.drop('ID_hd', axis=1, inplace=True)
            self.total_seq.append(seq.values.tolist())

        self.total_seq = np.array([np.array(i) for i in self.total_seq])
        return np.asarray(self.total_seq)

def make_data():
    path ='/home/jayeon/Documents/code/Hemodialysis/data' #raw_data
    files = ['Hemodialysis1_1007.csv','Hemodialysis2_1007.csv']
    # files = ['sample.csv']
<<<<<<< HEAD:src/csv_parser.py
    dataset = HemodialysisDataset(path,files,'MLP', save=True)
=======
    dataset = HemodialysisDataset(path,files,'RNN', save=True)
>>>>>>> feat: make rnn dataset & model:csv_parser.py
    return dataset


dataset = make_data()
