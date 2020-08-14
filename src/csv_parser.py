from torch.utils.data import Dataset
import pandas as pd
import os
import datetime as dt
import numpy as np
import torch
import json
import utils

stratified = False
EF_exp = False
remove_EF = False

class HemodialysisDataset():
    """Hemodialysis dataset from SNU"""

    def __init__(self, root_dir, csv_files, model_type, save=True, load=False):
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
        self.sbp_diff = -20.0
        self.dbp_diff = -10.0
        self.seed= 212
        self.load = load
        self._init_dataset(save)


    def __len__(self):
        return len(self.hemodialysis_frame)

    def __getitem__(self, idx):
        return self.hemodialysis_frame[idx]

    def _init_dataset(self, save):
        for f in self.files:
            tmp = pd.read_csv(os.path.join(self.root_dir, f), header=0)
            self.hemodialysis_frame = pd.concat([self.hemodialysis_frame, tmp], ignore_index=True)
        if self.load:
            self.hemodialysis_frame = self.hemodialysis_frame.loc[:1000]
        print('init', self.hemodialysis_frame.shape)
        
        if not self.load:
            if stratified:
                self.split_train_test()
            self.refine_dataset()
            print('after refine', self.hemodialysis_frame.shape)
            self.normalize()
            # self.hemodialysis_frame = self.hemodialysis_frame.loc[self.hemodialysis_frame.ID_class == self.training_type]
            if self.model_type == 'MLP':
                self.concat_history()
            else:
                self.make_sequence()
                print('after make_sequence', self.hemodialysis_frame.shape)
                self.add_pre_hemodialysis()
                print('after add_pre_hemodialysis', self.hemodialysis_frame.shape)
                self.order_target_column()
                print('after order_target_column', self.hemodialysis_frame.shape)
            if not stratified :
                self.split_train_test() # original
            self.columns = list(filter(lambda x: x not in ['ID_class', 'ID_hd'], self.hemodialysis_frame.columns))
            print('columns', self.hemodialysis_frame.columns)
            
            self.hemodialysis_frame.loc[:1500].to_csv('../data/tensor_data/{}/prerpoc.csv'.format(self.model_type), index=False)

        if save:
            if not self.load:
                print("Saving...")
                with open('../data/tensor_data/{}/mean_value.json'.format(self.model_type), 'w') as f:
                    f.write(json.dumps(self.mean_for_normalize))

                with open('../data/tensor_data/{}/std_value.json'.format(self.model_type), 'w') as f:
                    f.write(json.dumps(self.std_for_normalize))

                with open('../data/tensor_data/{}/columns.csv'.format(self.model_type), 'w') as f:
                    for c in self.columns:
                        f.write('%s\n' % c)
            for type in ['Train', 'Validation', 'Test']:
                df = self.hemodialysis_frame.loc[self.hemodialysis_frame.ID_class == type].copy()
                df = df.drop('ID_class', axis=1)
                if self.model_type == 'MLP':
                    df = df.drop('ID_hd', axis=1)
                    df = np.array(df).astype('float')
                else:
                    print(type, end=' ')
                    print(df.shape, end=' ')
                    
                    df = self.convert_to_sequence(df, type)
                    print(df.shape, end=' ')
                if not(os.path.isdir('../data/tensor_data/RNN')):
                    os.makedirs(os.path.join('../data/tensor_data/RNN'))
                # torch.save(df, '../tensor_data/RNN/{}.pt'.format(type))
                torch.save(df, '../data/tensor_data/{}/{}.pt'.format(self.model_type, type))

        else:
            print('Check.')
            self.hemodialysis_frame.drop('ID_class', axis=1, inplace=True)
            if self.model_type == 'MLP':
                np.array(self.hemodialysis_frame).astype('float')
            else:
                self.hemodialysis_frame = self.convert_to_sequence(self.hemodialysis_frame, 'Train')


    def split_train_test(self):
        if stratified :
            print('Splitting... stratified')
            import pickle
            with open("../data/raw_data/train_idx.txt", "rb") as fp:   # Unpickling
                train_idx = pickle.load(fp)
            with open("../data/raw_data/val_idx.txt", "rb") as fp:   # Unpickling
                val_idx = pickle.load(fp)
            with open("../data/raw_data/test_idx.txt", "rb") as fp:   # Unpickling
                test_idx = pickle.load(fp)
            print(sorted(train_idx)[:10])
            print(sorted(val_idx)[:10])
            print(sorted(test_idx)[:10])
            self.hemodialysis_frame.loc[:, 'ID_class'] = '0'
            self.hemodialysis_frame.loc[train_idx, 'ID_class'] = 'Train'
            self.hemodialysis_frame.loc[val_idx, 'ID_class'] = 'Validation'
            self.hemodialysis_frame.loc[test_idx, 'ID_class'] = 'Test'
            
                    
            print(len(train_idx), len(val_idx), len(test_idx))
        else:
            print('Splitting... no stratified')
            np.random.seed(self.seed)
            key = self.hemodialysis_frame['ID_hd'].unique()
            # idx = np.random.permutation(range(len(key)))
            # train_split, val_split = int(np.floor(0.7 * len(key))), int(np.floor(0.8 * len(key)))
            idx = range(len(key))
            train_split, val_split = int(np.floor(1.0 * len(key))), int(np.floor(1.0 * len(key)))
            train_idx, val_idx, test_idx = idx[:train_split], idx[train_split:val_split], idx[val_split:]
            if (not EF_exp) and (remove_EF) :
                self.hemodialysis_frame.drop(labels=['EF'], axis=1, inplace=True)
            print(len(train_idx), len(val_idx), len(test_idx))

            training_type = pd.DataFrame(data=key, columns=['ID_hd'])
            training_type['ID_class'] = 0; training_type.loc[train_idx, 'ID_class'] = 'Train'; #training_type.loc[val_idx, 'ID_class'] = 'Validation'; training_type.loc[test_idx, 'ID_class'] = 'Test'
            self.hemodialysis_frame = training_type.merge(self.hemodialysis_frame, how='inner', left_on=["ID_hd"], right_on=["ID_hd"])
        


    def refine_dataset(self):
        print('refine.....')
        self.hemodialysis_frame['Pt_id'] = [x.replace('*', '') for x in self.hemodialysis_frame['Pt_id']]
        self.hemodialysis_frame['Pt_id'] = self.hemodialysis_frame['Pt_id'].astype(str).astype(int)
        self.hemodialysis_frame['ID_hd'] = [x.replace('*', '') for x in self.hemodialysis_frame['ID_hd']]
        self.hemodialysis_frame['ID_hd'] = self.hemodialysis_frame['ID_hd'].astype(str).astype(int)
        categorical = ['HD_type','HD_acces','HD_prim','HD_dialysate','HD_dialyzer']
        # categorical = ['Pt_sex',  'HD_fut', 'CL_adm', 'CL_dm', 'CL_htn', 'CL_cad', 'CL_donor', 'CL_recipient', \
        #     'MED_bb', 'MED_ccb', 'MED_aceiarb', 'MED_spirono', 'MED_lasix', 'MED_statin', 'MED_minox', 'MED_aspirin', 'MED_plavix', 'MED_warfarin', \
        #         'MED_oha', 'MED_insulin', 'MED_allop', 'MED_febuxo', 'MED_epo', 'MED_pbindca', 'MED_pbindnoca']
        drop_columns = ['VS_rr', 'Time']
        self.hemodialysis_frame.drop(labels=drop_columns, inplace=True, axis=1)
        timestamp = pd.to_datetime(self.hemodialysis_frame['HD_duration']).dt
        self.hemodialysis_frame['HD_duration'] = timestamp.hour * 60 + timestamp.minute
        if EF_exp:
            self.hemodialysis_frame = self.hemodialysis_frame[~self.hemodialysis_frame.EF.isnull()] # EF 변수 없는 환자 drop
            if remove_EF:
                self.hemodialysis_frame.drop(labels=['EF'], axis=1, inplace=True)
            else:
                pass
        else :
            self.hemodialysis_frame.drop(labels=['EF'], axis=1, inplace=True)
            
        self.hemodialysis_frame.fillna(method='ffill', inplace=True)
        self.hemodialysis_frame['Pt_sex'] = self.hemodialysis_frame['Pt_sex'].replace({'M':0, 'F':1})
        # import pickle
        # with open("../data/stratified_w_EF/train_idx.txt", "rb") as fp:   # Unpickling
        #     train_idx = pickle.load(fp)
        # with open("../data/stratified_w_EF/half_val_idx.txt", "rb") as fp:   # Unpickling
        #     val_idx = pickle.load(fp)
        # with open("../data/stratified_w_EF/cal_idx.txt", "rb") as fp:   # Unpickling
        #     cal_idx = pickle.load(fp)
        # with open("../data/stratified_w_EF/test_idx.txt", "rb") as fp:   # Unpickling
        #     test_idx = pickle.load(fp)
        
        # for c in categorical:
        #     self.hemodialysis_frame[c] = [s.strip().lower() for s in self.hemodialysis_frame[c]]
        #     print('\ncate:{}\ntrain'.format(c),len((self.hemodialysis_frame.iloc[train_idx])[c].unique()) ,sorted((self.hemodialysis_frame.iloc[train_idx])[c].unique()))
        #     print('val',len((self.hemodialysis_frame.iloc[val_idx])[c].unique()), sorted((self.hemodialysis_frame.iloc[val_idx])[c].unique()))
        #     print('cal',len((self.hemodialysis_frame.iloc[cal_idx])[c].unique()), sorted((self.hemodialysis_frame.iloc[cal_idx])[c].unique()))
        #     print('test',len((self.hemodialysis_frame.iloc[test_idx])[c].unique()), sorted((self.hemodialysis_frame.iloc[test_idx])[c].unique()))
        #     print('test',len(self.hemodialysis_frame[c].unique()), sorted(self.hemodialysis_frame[c].unique()))
        # exit()
        self.hemodialysis_frame = pd.get_dummies(self.hemodialysis_frame, columns=categorical, prefix=categorical)
        self.hemodialysis_frame = self.hemodialysis_frame.loc[(self.hemodialysis_frame.VS_sbp > 0) & (self.hemodialysis_frame.VS_dbp > 0)]
        print('refine.....finished')

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
        # EF 변수 추가
        if (not EF_exp) or (remove_EF) :
            numerical_col = ['Pt_age', 'HD_duration', 'HD_ntime', 'HD_ctime', 'HD_prewt', 'HD_uf', 'VS_sbp', 'VS_dbp', 'VS_hr', 'VS_bt', 'VS_bfr', 'VS_uft', 'Lab_wbc', 'Lab_hb', 'Lab_plt', 'Lab_chol', 'Lab_alb', 'Lab_glu', 'Lab_ca', 'Lab_phos', 'Lab_ua', 'Lab_bun', 'Lab_scr', 'Lab_na', 'Lab_k', 'Lab_cl', 'Lab_co2']
        else:
            numerical_col = ['EF', 'Pt_age', 'HD_duration', 'HD_ntime', 'HD_ctime', 'HD_prewt', 'HD_uf', 'VS_sbp', 'VS_dbp', 'VS_hr', 'VS_bt', 'VS_bfr', 'VS_uft', 'Lab_wbc', 'Lab_hb', 'Lab_plt', 'Lab_chol', 'Lab_alb', 'Lab_glu', 'Lab_ca', 'Lab_phos', 'Lab_ua', 'Lab_bun', 'Lab_scr', 'Lab_na', 'Lab_k', 'Lab_cl', 'Lab_co2']
        
        for col in numerical_col:
            self.mean_for_normalize[col] = self.hemodialysis_frame[col].mean()
            self.std_for_normalize[col] = self.hemodialysis_frame[col].std()
            # if self.std_for_normalize[col] > 0:
            #     self.hemodialysis_frame[col] = (self.hemodialysis_frame[col] - self.mean_for_normalize[col]) / self.std_for_normalize[col]
            # else:
            #     self.hemodialysis_frame[col] = 0
        print('Normalizing... finished')

    def add_target_class(self):
        def eval_target(diff, type):
            if type == 'sbp':
                if diff < -20 :
                    return 0
                elif diff < -10 :
                    return 1
                elif diff < -5 :
                    return 2
                elif diff < 5 :
                    return 3
                elif diff < 10:
                    return 4
                elif diff < 20:
                    return 5
                else:
                    return 6

            if type == 'dbp':
                if diff < -10:
                    return 0
                elif diff < -5:
                    return 1
                elif diff < 5:
                    return 2
                elif diff < 9:
                    return 3
                else:
                    return 4

        self.hemodialysis_frame['VS_sbp_target_class'] = ((self.hemodialysis_frame['VS_sbp_target'] - self.hemodialysis_frame['VS_sbp']) * self.std_for_normalize['VS_sbp']).apply(lambda x: eval_target(x,'sbp'))
        self.hemodialysis_frame['VS_dbp_target_class'] = ((self.hemodialysis_frame['VS_dbp_target'] - self.hemodialysis_frame['VS_dbp']) * self.std_for_normalize['VS_dbp']).apply(lambda x: eval_target(x,'dbp'))


    def add_pre_hemodialysis(self):
        min_bp = self.hemodialysis_frame.groupby(['Pt_id', 'ID_hd'])['VS_sbp', 'VS_dbp'].min().reset_index()
        max_bp = self.hemodialysis_frame.groupby(['Pt_id', 'ID_hd'])['VS_sbp', 'VS_dbp'].max().reset_index()
        init_bp = self.hemodialysis_frame.loc[self.hemodialysis_frame.HD_ctime_raw == 0][['Pt_id', 'ID_hd', 'VS_sbp', 'VS_dbp']]
        init_bp.columns = ['Pt_id', 'ID_hd', 'VS_sbp_init_in_pre_hd', 'VS_dbp_init_in_pre_hd']
        pre_hd = min_bp.merge(max_bp, how='inner', on=['Pt_id', 'ID_hd'], suffixes=('_min_in_pre_hd', '_max_in_pre_hd'))
        pre_hd = init_bp.merge(pre_hd, how='inner', on=['Pt_id', 'ID_hd'])
        pre_hd['rank'] = pre_hd.sort_values(['Pt_id', 'ID_hd'], ascending=[True, True]).groupby(['Pt_id']).cumcount() + 1
        self.hemodialysis_frame = self.hemodialysis_frame.merge(pre_hd[['Pt_id', 'ID_hd', 'rank']],on=['Pt_id', 'ID_hd'], how='inner')
        self.hemodialysis_frame['rank'] -= 1
        self.hemodialysis_frame = self.hemodialysis_frame.merge(pre_hd, on=['Pt_id', 'rank'], how='left', suffixes=('', '_pre'))
        self.hemodialysis_frame['pre_hd'] = [0 if np.isnan(x) else 1 for x in self.hemodialysis_frame['VS_sbp_min_in_pre_hd']]
        self.hemodialysis_frame.drop(labels=['Pt_id', 'rank', 'ID_hd_pre', 'HD_ctime_raw'], axis=1, inplace=True)
        self.hemodialysis_frame.fillna(0.0, inplace=True)


    def order_target_column(self):
        target_columns = ['VS_sbp_target', 'VS_dbp_target', 'VS_sbp_target_class', 'VS_dbp_target_class']
        input_columns = self.hemodialysis_frame.columns[
            [False if x in target_columns else True for x in self.hemodialysis_frame.columns]].to_list()
        self.hemodialysis_frame = self.hemodialysis_frame[input_columns + target_columns]


    def convert_to_sequence(self, df, type):
        if False:  grouped = df.groupby('ID_hd')
        else:     grouped = df.sort_values(['ID_hd', 'HD_ctime'], ascending=[True,True]).groupby('ID_hd')
        unique = df['ID_hd'].unique()
        self.total_seq = []
        for id_ in unique:
            seq = grouped.get_group(id_)  # dataframe type
            seq.drop('ID_hd', axis=1, inplace=True)
            self.total_seq.append(seq.values.tolist())

        self.total_seq = [np.array(i) for i in self.total_seq]
        self.total_seq = data_modify_same_ntime(np.array(self.total_seq), 60, type, '../data/tensor_data/RNN/', False)

        return self.total_seq



def data_modify_same_ntime(data, ntime=60, d_type='Train', base_dir=None, mask=True):
   new_data = np.copy(data)
   
   if base_dir is None:
       mean_HD_ctime = 112.71313384332716
       std_HD_ctime = 78.88638128125473
       mean_VS_sbp = 132.28494659660691
       mean_VS_dbp = 72.38785072807198
       std_VS_sbp = 26.863242507719363
       std_VS_dbp = 14.179094454260184
       std_HD_ntime = 24.563710661941634
       mean_HD_ntime = 31.632980458822722
       idx_HD_ctime = 6
       idx_VS_sbp = 11
       idx_VS_dbp = 12
       idx_target_ntime = -12
   else:
        with open(os.path.join(base_dir, 'mean_value.json')) as mean, open(os.path.join(base_dir, 'std_value.json')) as std, open(os.path.join(base_dir, 'columns.csv')) as columns:
           mean_data = json.load(mean)
           std_data = json.load(std)
           mean_HD_ctime = mean_data['HD_ctime']
           mean_VS_sbp = mean_data['VS_sbp']
           mean_VS_dbp = mean_data['VS_dbp']
           std_HD_ctime = std_data['HD_ctime']
           std_VS_sbp = std_data['VS_sbp']
           std_VS_dbp = std_data['VS_dbp']
           for idx, col in enumerate(columns.readlines()):
               col = col.strip()
               if col == 'VS_sbp':
                   idx_VS_sbp = idx
               elif col == 'VS_dbp':
                   idx_VS_dbp = idx
               elif col == 'HD_ctime':
                   idx_HD_ctime = idx
        std_HD_ntime = 24.563710661941634
        mean_HD_ntime = 31.632980458822722
        idx_target_ntime = -12

#    c_time_list = [(data[i][:,idx_HD_ctime]*std_HD_ctime+mean_HD_ctime).astype(int) for i in range(len(data))]
#    sbp_list =[(data[i][:,idx_VS_sbp]*std_VS_sbp+mean_VS_sbp).astype(int) for i in range(len(data))]
#    map_list =[((data[i][:,idx_VS_dbp]*std_VS_dbp+mean_VS_dbp)*2/3. + (data[i][:,idx_VS_sbp]*std_VS_sbp+mean_VS_sbp)/3.).astype(int) for i in range(len(data))]
#    target_n_time_list = [(data[i][:,idx_target_ntime]*std_HD_ntime+mean_HD_ntime).astype(int) for i in range(len(data))]
   c_time_list = [(data[i][:,idx_HD_ctime]).astype(int) for i in range(len(data))]
   sbp_list =[(data[i][:,idx_VS_sbp]).astype(float) for i in range(len(data))]
   map_list =[((data[i][:,idx_VS_dbp])*2./3. + (data[i][:,idx_VS_sbp])/3.).astype(float) for i in range(len(data))]
   target_n_time_list = [(data[i][:,idx_target_ntime]).astype(int) for i in range(len(data))]
   # 각 frame의 c_time, sbp, map를 받았음.
   # <<<<중요>>>>> 7,12,13은 data 형태에 따라서 바꿔줘야 함

   for data_idx in range(len(data)):
        last_target_sbp = (data[data_idx][-1,-4]).astype(np.float)
        last_target_map = ((data[data_idx][-1,-3])*2./3. + (data[data_idx][-1,-4])/3.).astype(np.float)
    #    print(c_time_list[data_idx], target_n_time_list[data_idx])
        c_time_list[data_idx] = np.insert(c_time_list[data_idx], len(c_time_list[data_idx]), c_time_list[data_idx][-1]+target_n_time_list[data_idx][-1])
        sbp_absolute_value = sbp_list[data_idx].astype(np.float)
        map_absolute_value = map_list[data_idx].astype(np.float)
        sbp_absolute_value = np.insert(sbp_absolute_value, len(sbp_absolute_value), last_target_sbp).astype(np.float)
        map_absolute_value = np.insert(map_absolute_value, len(map_absolute_value), last_target_map).astype(np.float)

        sbp_init_value = sbp_list[data_idx][0].astype(np.float)
        map_init_value = map_list[data_idx][0].astype(np.float)

        sbp_diff = (sbp_absolute_value.astype(np.float) - sbp_init_value.astype(np.float)).astype(np.float)
        map_diff = (map_absolute_value.astype(np.float) - map_init_value.astype(np.float)).astype(np.float)

        temp_data_concat = np.zeros((len(sbp_diff)-1, 5))

        for frame_idx in range(len(data[data_idx])):
            criterion_flag = (c_time_list[data_idx][frame_idx] + ntime >= c_time_list[data_idx]) & (c_time_list[data_idx][frame_idx] < c_time_list[data_idx]) # shape : [True, True, True, True, False, False, False, ....]
            
            if np.sum(((sbp_absolute_value[criterion_flag] - sbp_absolute_value[frame_idx]) <= -20.0+1e-5).astype(int)) : curr_sbp_exist = 1
            else : curr_sbp_exist = 0
            if np.sum(((map_absolute_value[criterion_flag] - map_absolute_value[frame_idx]) <= -10.0+1e-5).astype(int)) : curr_map_exist = 1
            else : curr_map_exist = 0

            if np.sum((sbp_diff[criterion_flag]<=-20.0+1e-5).astype(int)) : sbp_exist = 1    # 초기값 대비 20 이상 떨어지는게 존재하면 1, else 0.
            else : sbp_exist = 0
            if np.sum((map_diff[criterion_flag]<=-10+1e-5).astype(int)) : map_exist = 1
            else : map_exist = 0
            if np.sum((sbp_absolute_value[criterion_flag]<90.0).astype(int)) : sbp_under_90 = 1
            else : sbp_under_90 = 0

            temp_data_concat[frame_idx] = np.array((sbp_exist, map_exist, sbp_under_90, curr_sbp_exist, curr_map_exist))

        new_data[data_idx] = np.concatenate((new_data[data_idx], temp_data_concat), axis=1)


   # torch.save(data, 'data/tensor_data/1227_EF_60min/{}_{}min.pt'.format(d_type, ntime)) # save root 잘 지정해줄 것
   return new_data


def make_data():
    # path ='../data/raw_data/'
    # files = ['Hemodialysis1_0122.csv','Hemodialysis2_0122.csv']
    # dataset = HemodialysisDataset(path, files, 'RNN_tmp', save=True)

    path ='../data/tensor_data/RNN_tmp/'
    files = ['prerpoc.csv']
    dataset = HemodialysisDataset(path, files, 'RNN_tmp', save=False, load=True)
    
    return dataset


dataset = make_data()
