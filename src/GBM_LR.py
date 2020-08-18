import lightgbm as lgb
import torch
import numpy as np

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


num_random_seed = 58
print(num_random_seed)
data_root = '../../data/raw_data/0813/pt_file_{}/wo_EF/'.format(num_random_seed)

Load_dir = None
Save_dir = '../fig/LightGBM/'

target_name_list = ['IDH1','IDH2','IDH3','IDH4','IDH5']


param = {'num_leaves': 31, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 10000


train = []
train = torch.load('{}/Train.pt'.format(data_root))
train= [seq for row in train for seq in row]
train = np.array(train)
# train = np.concatenate([train[:,:26], train[:,29:31], train[:,36:]],axis=1)
# train = np.concatenate([train[:,:31], train[:,36:]],axis=1)
train = np.concatenate([train[:,3:5], train[:,5:31], train[:,36:]],axis=1)



valid = torch.load('{}/Validation.pt'.format(data_root))
valid = [seq for row in valid for seq in row]
valid = np.array(valid)
# valid = np.concatenate([valid[:,:26], valid[:,29:31], valid[:,36:]],axis=1)
valid = np.concatenate([valid[:,3:5], valid[:,5:31], valid[:,36:]],axis=1)

test = torch.load('{}/Test.pt'.format(data_root))
test = [seq for row in test for seq in row]
test = np.array(test)
# test = np.concatenate([test[:,:26], test[:,29:31], test[:,36:]],axis=1)
test = np.concatenate([test[:,3:5], test[:,5:31], test[:,36:]],axis=1)

print(train.shape, valid.shape, test.shape)


import json
with open('{}/column_list.json'.format(data_root), "r") as f:
    features_names = json.load(f)
print(features_names, len(features_names))
features_names = features_names[:-5]
print(features_names, len(features_names))

# features_names = features_names[:26] + features_names[29:31] + features_names[36:]
# features_names = features_names[:31] + features_names[36:]
features_names = features_names[3:5] + features_names[5:31] + features_names[36:]
print(features_names, len(features_names))


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features = scaler.fit_transform(train[:,:-5])
valid_features = scaler.transform(valid[:,:-5])
test_features = scaler.transform(test[:,:-5])



target_idx_dict = {'IDH1':-5, 'IDH2':-4, 'IDH3':-3, 'IDH4':-2, 'IDH5':-1}


for target_name in target_name_list:
    train_data = lgb.Dataset(train_features, label=train[:,target_idx_dict[target_name]], feature_name=features_names)
    valid_data = lgb.Dataset(valid_features, label=valid[:,target_idx_dict[target_name]], feature_name=features_names)


    test_target =test[:,target_idx_dict[target_name]]

    import os
    if not os.path.exists(Save_dir):
        os.mkdir(Save_dir)



    lgb_model = lgb.train(param, train_data, num_round, valid_sets=[valid_data], early_stopping_rounds=5, verbose_eval=False)
    lgb_model.save_model('{}/models/LightGBM_{}.pkl'.format(Save_dir, target_name))



    # feature importance
    importance_save_root = '{}/importance/'.format(Save_dir)
    if not os.path.exists(importance_save_root):
        os.mkdir(importance_save_root)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22})
    ax = lgb.plot_importance(lgb_model, figsize=(40,40), dpi=800)
    ax.set_title('')
    plt.savefig('{}/LightGBM_{}.pdf'.format(importance_save_root, target_name))
    # plt.savefig('{}/LightGBM_{}.jpg'.format(importance_save_root, target_name))
    plt.close()

    pred_s = lgb_model.predict(test_features)
    average_precision = average_precision_score(test_target, pred_s)
    roc_fpr_array, roc_tpr_array, roc_thresholds = metrics.roc_curve(test_target, pred_s)
    pr_precision_array, pr_recall_array, pr_thresholds = metrics.precision_recall_curve(test_target, pred_s)
    auc = metrics.auc(roc_fpr_array, roc_tpr_array)

    print('GBM {} :  AUROC {:.2f} / AUPRC {:.2f}'.format(target_name,auc*100., average_precision*100.))


    with open('{}/conf/conf_{}.txt'.format(Save_dir,target_name), 'w') as f:
        for idx, pred in enumerate(pred_s):
    #         print(idx, pred)
            f.write("{}\t{}\n".format(str(pred), str(test_target[idx])))



    log_clf = LogisticRegression() 
    log_clf.fit(train_features,train[:,target_idx_dict[target_name]]) 
    log_clf.score(valid_features, valid[:,target_idx_dict[target_name]])

    import pickle
    filename = '{}/models/LogiReg_{}.pkl'.format(Save_dir, target_name)
    pickle.dump(log_clf, open(filename, 'wb'))


    pred_s = log_clf.predict_proba(test_features)[:,1]

    with open('../fig/LogisticRegression/conf/conf_{}.txt'.format(target_name), 'w') as f:
        for idx, pred in enumerate(pred_s):
    #         print(idx, pred)
            f.write("{}\t{}\n".format(str(pred), str(test_target[idx])))


    average_precision = average_precision_score(test_target, pred_s)
    roc_fpr_array, roc_tpr_array, roc_thresholds = metrics.roc_curve(test_target, pred_s)
    pr_precision_array, pr_recall_array, pr_thresholds = metrics.precision_recall_curve(test_target, pred_s)
    auc = metrics.auc(roc_fpr_array, roc_tpr_array)

    print('LR {} :  AUROC {:.2f} / AUPRC {:.2f}'.format(target_name,auc*100., average_precision*100.))

    import numpy as np
    pred = np.random.uniform(0,1,len(test_target))

    average_precision = average_precision_score(test_target, pred)
    roc_fpr_array, roc_tpr_array, roc_thresholds = metrics.roc_curve(test_target, pred)
    pr_precision_array, pr_recall_array, pr_thresholds = metrics.precision_recall_curve(test_target, pred)
    auc = metrics.auc(roc_fpr_array, roc_tpr_array)

    print('Random {} :  AUROC {:.2f} / AUPRC {:.2f}'.format(target_name,auc*100., average_precision*100.))