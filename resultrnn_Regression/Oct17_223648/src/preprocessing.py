#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # TODO: 
# 09.09.19
#  - ~~Drop 'Patient Outcome' column~~
#  - ~~Drop raw data that have revised data~~
#  - ~~Identity data types (Numerical or Categorical)~~
#  
# 09.18.19
#  - ~~Identity feature types (HD,VS,CL,Lab)~~
#  - ~~Concat HD_ntim onto t-1 along with target (VS_sbp, VS_dbp)~~
#  - Data Analysis
#  - ~~Convert to Tensor~~

# In[2]:


PATH = ['Hemodialysis1.csv','Hemodialysis2.csv']
df_ = pd.DataFrame()
for i in PATH:
    temp_df = pd.read_csv(i, engine='c', header=0)
    df_ = pd.concat([temp_df,df_])


# In[3]:


df = df_.copy()


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.columns[df.isnull().any()]


# In[7]:


df[df.columns[df.isnull().any()]].isnull().sum() / len(df)


# In[8]:


# 결측치가 17% 인 VS_rr 제거 
df.drop(labels=['VS_rr'], inplace=True, axis=1)

# 환자 Id 정보 제거 
df.drop(labels='Pt_id', inplace=True, axis=1)


# In[9]:


df.info()


# In[10]:


df.fillna(method='ffill', inplace=True) # forward filling 으로 처리 
df.columns[df.isnull().any()]


# ### Data type 분류

# In[49]:




#non-int, non-float 중에서 categorical type 선정 
categorical = ['HD_type','HD_acces','HD_prim','HD_dialysate','HD_dialyzer']

#Target 으로 하는 feature 선정
target_col = ['VS_sbp','VS_dbp']

#Metadata 
id_col = ['ID_hd', 'ID_timeline', 'ID_class']

#나머지 features 
patient_col = ['Pt_sex', 'Pt_age'] #Pt_id 는 제외 
hemo_col = ['HD_type', 'HD_duration', 'HD_ntime', 'HD_ctime', 'HD_acces',
       'HD_prewt', 'HD_uf', 'HD_hep', 'HD_fut', 'HD_prim', 'HD_dialysate','HD_dialyzer']
vs_col = ['VS_sbp', 'VS_dbp', 'VS_hr', 'VS_rr', 'VS_bt', 'VS_bfr',
       'VS_uft']
cl_col = ['CL_adm', 'CL_dm', 'CL_htn', 'CL_cad', 'CL_donor',
       'CL_recipient']
lab_col = ['Lab_wbc', 'Lab_hb', 'Lab_plt', 'Lab_chol', 'Lab_alb',
       'Lab_glu', 'Lab_ca', 'Lab_phos', 'Lab_ua', 'Lab_bun', 'Lab_scr',
       'Lab_na', 'Lab_k', 'Lab_cl', 'Lab_co2']
med_col = ['MED_bb', 'MED_ccb',
       'MED_aceiarb', 'MED_spirono', 'MED_lasix', 'MED_statin', 'MED_minox',
       'MED_aspirin', 'MED_plavix', 'MED_warfarin', 'MED_oha', 'MED_insulin',
       'MED_allop', 'MED_febuxo', 'MED_epo', 'MED_pbindca', 'MED_pbindnoca']


# In[50]:


for i in categorical:
    print(df[i].unique())


# ## Categorical Feature Encoding

# In[51]:


total_df = pd.get_dummies(df,columns=categorical, prefix=categorical) 
total_df.head(5)


# In[52]:


total_df['Pt_sex'] = total_df['Pt_sex'].replace({'M':0, 'F':1})


# ## TODO: Data Analysis (Visualization)

# In[ ]:





# In[ ]:





# ## Input/Target Data 생성

# In[53]:


#각 투석건의 타임프레임 개수 
from ordered_set import OrderedSet
from collections import OrderedDict

h_ID = OrderedSet(df['ID_hd']) #hemodialysis ID

def hID_count(df,h_ID):
    hID_dict = OrderedDict(zip(h_ID,[0 for i in range(len(h_ID))]))
    for row in df['ID_hd']:
        hID_dict[row] += 1 
    return hID_dict


# In[54]:


hID_dict = hID_count(df,h_ID)


# In[55]:


# 투석 건수 
len(hID_dict)


# In[56]:


# Concat. 되는 feature 들 정의 
features = [i for i in list(df.columns) if i not in patient_col + id_col+ target_col+categorical]


# In[57]:


"""
Window_size = 2 의 df 를 concat 하여 하나의 input data 생성 
e.g. 8개의 타임프레임에서 총 6개의 데이터 생성
타겟값은 :
        1) i+2 시점의 vital signal (VS_sbp, VS_dbp) 
        2) i+2 시점의 vital signal 상승/저하 여부 
"""

import time
from ordered_set import OrderedSet

def eval_target(target_df,input_df,rule=None):
    #TODO: rule 명시
    if target_df['VS_sbp'] >= input_df['VS_sbp']+5:
        return True 
    
def sliding_window(h_ID, features=features, total_df=total_df, window_size=2):
    """
    Multiprocessing 용
    Inputs : 
        h_ID : 투석 건 ID
        total_df : 생성된 데이터프레임 
    Outputs:
        각 투석 건에 대해서 생성된 데이터 프레임. 생성한 뒤 모두 concat 하여 최종 데이터프레임 생성 
    """
    df = total_df.loc[total_df['ID_hd']==h_ID] 
    
    start_idx = 0 
    end_idx = len(df)
    new_df = pd.DataFrame()

    for ind in range(start_idx,end_idx-window_size):
        next_window_feature = df.iloc[ind+1][features].rename(lambda x: x+'2') # concat features
        next_window_feature['pred_time'] = df.iloc[ind+2]['HD_ntime'] # 예측하고자 하는 시점이 몇 분 뒤인지 feature 추가
        next_window_target = df.iloc[ind+1][target_col].rename(lambda x: x+'_target') # concat regression target 
        next_window_target['clas_target'] = 1 if eval_target(df.iloc[ind+2], df.iloc[ind+1]) else 0 # concat classification target
        new_df = new_df.append(pd.concat([df.iloc[ind], next_window_feature, next_window_target], axis = 0), ignore_index=True)
    
    return new_df 



#Single processing 사용시 예시; Input : total df 
# def sliding_window_df(total_df, features,window_size=2):
#     global hID_count, h_ID
    
#     start_time = time.time()

#     pid = OrderedSet(df['ID_hd'])
#     hID_dict = hID_count(df,h_ID)
#     print("ID List Created \n Ready to Iterate through Data")
#     print('***' * 10)

#     new_df = pd.DataFrame()
#     start_idx = 0
#     end_idx = 0 
#     cnt = 0

#     for (pid, count) in hID_dict.items():
#         end_idx += count
#         print("------On {} ID, Time Duration : {}".format(cnt, time.time()-start_time))
# #         if cnt == 1000:
# #             return new_df 
#         for ind in range(start_idx,end_idx-window_size):
#             next_window_feature = df.iloc[ind+1][features].rename(lambda x: x+'2')
#             next_window_target = df.iloc[ind+1][target_col].rename(lambda x: x+'_target')
#             next_window_target['clas_target'] = 1 if eval_target(df.iloc[ind+2], df.iloc[ind+1]) else 0 
#             new_df = new_df.append(pd.concat([df.iloc[ind], next_window_feature, next_window_target], axis = 0), ignore_index=True)

#         start_idx += count
#         cnt += 1

#     print('\n\n Total Time Elapsed : {}'.format(time.time()- start_time))

#     return new_df
    


# In[59]:


from multiprocessing import Pool
import time
import warnings
warnings.filterwarnings(action='once')

start_time = time.time()

pool = Pool(7) # Processor 7 개 사용 
new_df = pd.concat(pool.map(sliding_window, list(hID_dict)[:10]))
pool.close()
pool.join()

print('Time elapsed: ', time.time()-start_time)


# In[60]:


#Check for any null values
new_df[new_df.columns[new_df.isnull().any()]].isnull().sum() / len(new_df)


# In[61]:


new_df.head(5)
# Columns 개수 = 136 (기존 total_df) + 49 (features) + 2 (regression target) + 1 (clas_target) + 1 (pred_time_differnece)


# ### 처리 속도 기록 
# #### 09.17일 (Single process)
# - 11시간 : 45241 개 
# - 7시간 : 35000개 
# - 3시간: 25000 개
# - 1시간: 15000개 
# - 12분: 5000개 

# ## TODO: Input Data Analysis 

# In[62]:


new_df.columns


# In[63]:


#Classification Data 
print("Length of Data", len(new_df))
print("Positive Target Ratio", new_df.loc[new_df['clas_target']==1, 'clas_target'].sum()/len(new_df))


# ## Save as json & Tensor 

# In[ ]:


new_df.to_json(path='0919_MLP.json')


# In[104]:


import torch.nn as nn
import torch


target_col_list = [i for i in new_df.columns if 'target' in i] 
target = new_df['VS_sbp_target']

y = torch.tensor(target.values, dtype=torch.float)
X = torch.tensor(new_df.drop(labels=id_col+target_col_list, axis=1).values, dtype=torch.float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[105]:


class ToyNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super(ToyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class ToyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.input = data[0]
        self.target = data[1]

    def __getitem__(self, idx):
        x, y = self.input[idx], self.target[idx]
        return (x,y)

    def __len__(self):
        return len(self.input)


# In[107]:


model = ToyNet(input_size, hidden_size).to(device)    
toydataset = ToyDataset((X,y))
train_loader = torch.utils.data.DataLoader(toydataset, batch_size=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train model
input_size = 183
hidden_size = 500
num_epochs = 10
batch_size = 20
learning_rate = 0.0005

total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0
    total = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs).float()
#         print(outputs.shape)
        targets = targets.float()
#         print(outputs.shape)

        loss = criterion(outputs, targets)
        total += inputs.size(0)
        running_loss += loss/total
        

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, running_loss))


# In[ ]:




