import loader
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib as mpl

# tableau20 = np.array(
#             ((31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229))) / 255

tableau20 = np.array(((44, 160, 44),(31, 119, 180), (255, 127, 14),  (214, 39, 40))) / 255

# Say, "the default sans-serif font is COMIC SANS"
# mpl.rcParams['font.DejaVu Sans'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
# mpl.rcParams['font.family'] = "Arial"
# print(mpl.font_manager.get_fontconfig_fonts())
# exit()
# mpl.rcParams["font.sans-serif"] = "Arial"# ["Arial", "Liberation Sans", "Bitstream Vera Sans"]
mpl.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def confidence_save_and_cal_auroc(conf_dict, target_dict, data_type, save_dir, epoch=9999, step=0, composite=True, cal_roc=True):
    
    save_dir += '/auroc'
    # category ={"under90":0, "Composite Outcome from Initial Timeframe":1, "Composite Outcome from Present Timeframe":2}
    auc_dict = dict()
    stat_dict = dict()
    ap_dict = dict()
    for key, value in conf_dict.items():
        print("Making {} roc curve...".format(key))
        key_dir = save_dir + '/' + key
        if not os.path.isdir(key_dir):
            os.makedirs(key_dir)
        if cal_roc :
            auroc, ap, stat_array = roc_curve_plot(key_dir, key, value, target_dict[key], save_dir=key_dir)
            auc_dict[key] = auroc
            stat_dict[key] = stat_array
            ap_dict[key] = ap

    return auc_dict, ap_dict, stat_dict

def roc_curve_plot(load_dir, category, conf, target, save_dir=None):
    # calculate the AUROC
    from sklearn import metrics
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    average_precision = average_precision_score(target, conf)
    roc_fpr_array, roc_tpr_array, roc_thresholds = metrics.roc_curve(target, conf)
    pr_precision_array, pr_recall_array, pr_thresholds = metrics.precision_recall_curve(target, conf)
    auc = metrics.auc(roc_fpr_array, roc_tpr_array)

    fig = plt.figure(figsize=(10,5))
    if category == 'under90':
        fig.suptitle("Under 90", fontsize=16)
    elif category == 'init_composite' :
        fig.suptitle("composite from init timeframe", fontsize=16)
    elif category == 'curr_composite' :
        fig.suptitle("composite from present timeframe", fontsize=16)
    else:
        print('subfigure sub title error')
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(roc_fpr_array, roc_tpr_array,  linewidth=3 )
    ax1.axhline(y=1.0, color='black', linestyle='dashed')
    # ax1.set_title('ROC {} {}epoch'.format(category, epoch))
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel('FPR(False Positive Rate)')
    ax1.set_ylabel('TPR(True Positive Rate)')
    ax1.text(0.5,0.05, s='auroc : {:.5f}'.format(auc),  fontsize=15)

    ax2 = fig.add_subplot(1,2,2)
    # ax2.plot(tpr_list, prec_list,  linewidth=3 )
    ax2.plot(pr_recall_array, pr_precision_array,  linewidth=3 )
    ax2.axhline(y=1.0, color='black', linestyle='dashed')
    # ax2.set_title('PR_Curve {} {}epoch'.format(category, epoch))
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel('TPR(Recall)')
    ax2.set_ylabel('Precision')
    ax2.text(0.05,0.05, s='AP : {:.5f}'.format(average_precision),  fontsize=15)

    
    if save_dir is None:  # load dir에 저장
        fig.savefig('{}/ROC_{}.jpg'.format(load_dir, category), dpi=300)
        fig.savefig('{}/ROC_{}.png'.format(load_dir, category), dpi=300)
        fig.savefig('{}/ROC_{}.pdf'.format(load_dir, category), dpi=300)
    else:  # 다른 dir을 지정하여 저장
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open('{}/{}_roc_tpr.txt'.format(save_dir, category), 'w') as f1, open('{}/{}_roc_fpr.txt'.format(save_dir, category), 'w') as f2, open('{}/{}_pr_prec.txt'.format(save_dir, category), 'w') as f3, open('{}/{}_pr_recall.txt'.format(save_dir, category), 'w') as f4 :
            if len(roc_tpr_array) != len(roc_fpr_array):
                print('length error / roc')
                assert len(roc_tpr_array) != len(roc_fpr_array)
            if len(pr_precision_array) != len(pr_recall_array):
                print('length error  / pr')
                assert len(pr_precision_array) != len(pr_recall_array)
            for i in range(len(roc_tpr_array)):
                f1.write('{}\n'.format(roc_tpr_array[i]))
                f2.write('{}\n'.format(roc_fpr_array[i]))
            for i in range(len(pr_precision_array)):
                f3.write('{}\n'.format(pr_precision_array[i]))
                f4.write('{}\n'.format(pr_recall_array[i]))
        fig.savefig('{}/ROC_{}.jpg'.format(save_dir, category), dpi=300)
        fig.savefig('{}/ROC_{}.png'.format(save_dir, category), dpi=300)
        fig.savefig('{}/ROC_{}.pdf'.format(save_dir, category), dpi=300)
    plt.close("all")
    return auc, average_precision, [roc_tpr_array, roc_fpr_array, pr_precision_array, pr_recall_array]






def sklearn_calibration_histogram(conf_dict, target_dict, save_dir, method, nbins=20, cali_method=None):
    from sklearn.calibration import calibration_curve
    print('calibration sklearn')
    color_map = {'RNN':0, 'MLP':1, 'LightGBM':2, 'LogisticRegression':3}
    bin_size = [0.01, 0.04, 0.05, 0.10]
    return_xaxis_dict = dict()
    return_yaxis_dict = dict()
    for step in bin_size :
        return_xaxis_dict[step] = dict()
        return_yaxis_dict[step] = dict()
        for key, value in conf_dict.items():
            nbins = int(1.0 /  step)
            fraction_of_positives, mean_predicted_value = calibration_curve(target_dict[key], value, n_bins=nbins)
            return_yaxis_dict[step][key] = fraction_of_positives
            return_xaxis_dict[step][key] = mean_predicted_value
            bins_ = np.insert(mean_predicted_value.copy(), len(mean_predicted_value), 1.0)

            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            xaxis_font_size = 20
            yaxis_font_size = 20
            tick_font_size = 20
            legend_font= 20
            fig = plt.figure(figsize=(8, 8)) 
            ax = plt.subplot()
            plt.tight_layout(rect=[0.18, 0.1, 0.96, 0.94])
            
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", color='b', alpha=1.0, label='{} Calibration'.format(method), )
            ax.plot(bins_, bins_, "k:" ,color='black', alpha=1.0, linestyle=':', label='Ideal Calibration')
            
            # ax.plot(mean_predicted_value, fraction_of_positives, "s-", color=tableau20[color_map[method]], alpha=1.0, label='{}'.format(method), )
            

            ax.set_ylim([0.0, 1.00001])
            ax.set_xlim([0.0, 1.00001])
            ax.set_yticks((0.0,0.25,0.5,0.75,1.0))
            ax.set_xticks((0.0,0.25,0.5,0.75,1.0))
            ax.set_xlabel('Confidence', fontsize=xaxis_font_size)
            ax.set_ylabel('Fraction of positives', fontsize=yaxis_font_size)
            ax.yaxis.set_label_coords(-0.21, 0.50)
            ax.xaxis.set_label_coords(0.50, -0.12)
            ax.grid(color='black', linestyle=':', alpha=0.8)
            # ax.set_title('calibration_{}'.format(key))
            ax.tick_params(direction='out', length=5, labelsize=tick_font_size, width=4, grid_alpha=0.5)
            ax.legend(loc='upper left', fontsize=legend_font)

            if cali_method is None:
                if not os.path.isdir('{}/calibration/bin_size_{:.2f}/'.format(save_dir, step)):
                    os.makedirs('{}/calibration/bin_size_{:.2f}/'.format(save_dir, step))
                # if not os.path.isdir('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize)):
                #     os.makedirs('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize))
                plt.savefig('{}/calibration/bin_size_{:.2f}/Line_graph_{}_calibration.png'.format(save_dir, step, key))
                # plt.savefig('{}/calibration/bin_size_{}/jpg/{}_graph_{}_calibration.jpg'.format(save_dir, binsize, plot_type, key))
            else:
                if not os.path.isdir('{}/calibration/bin_size_{:.2f}_{}/'.format(save_dir, step, cali_method)):
                    os.makedirs('{}/calibration/bin_size_{:.2f}_{}/'.format(save_dir, step, cali_method))
                # if not os.path.isdir('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize)):
                #     os.makedirs('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize))
                plt.savefig('{}/calibration/bin_size_{:.2f}_{}/Line_graph_{}_calibration.png'.format(save_dir, step, key, cali_method))
                # plt.savefig('{}/calibration/bin_size_{}/jpg/{}_graph_{}_calibration.jpg'.format(save_dir, binsize, plot_type, key))
            plt.close()

    return return_yaxis_dict, return_xaxis_dict

def calibration_histogram(conf_dict, target_dict, save_dir, method, cali_method=None):
    # item_list = ['under90', 'init_sbp','init_map','curr_sbp','curr_map']
    bin_size = [0.01, 0.02, 0.03,0.04, 0.05, 0.06,0.10]
    ece_dict = dict()
    similar_ece_dict = dict()

    return_xaxis_dict = dict()
    return_yaxis_dict = dict()
    for step in bin_size :
        ece_dict[step] = dict()
        similar_ece_dict[step] = dict()
        return_xaxis_dict[step] = dict()
        return_yaxis_dict[step] = dict()
        for key, value in conf_dict.items():
            # confi_and_target = np.loadtxt('{}/conf/conf_{}.txt'.format(save_dir,item), delimiter=',', dtype=str)
            # confidences = np.asarray([float(str_conf_and_target[:-4]) for str_conf_and_target in sorted(confi_and_target)])
            # targets = np.asarray([int(float(str_conf_and_target[-3:])) for str_conf_and_target in sorted(confi_and_target)])

            confidences = value
            targets = target_dict[key]

            bins = np.arange(0.00, 1.000, step=step)
            ratio_list = list()
            data_count_list = list()
            data_ones_count_list = list()
            data_zeros_count_list = list()
            ece = 0.0
            similar_ece=0.0
            for delta in bins:
                where = np.where(((confidences < delta+step) & (confidences >= delta)))

                delta_targets = targets[where]
                ones_flag = (delta_targets == 1)
                zeros_flag = (delta_targets == 0)

                data_count_list.append(min(len(where[0]), 5000))
                data_ones_count_list.append(ones_flag.sum())
                data_zeros_count_list.append(zeros_flag.sum())


                if len(ones_flag) == 0:
                    ratio_list.append(0)
                    continue

                ratio = float(ones_flag.sum()) / float(len(ones_flag))
                ratio_list.append(ratio)
                
                ece_cal = (  (np.abs(ones_flag.sum() - confidences[where].sum() ))  )
                similar_ece_cal = (  (np.abs(ones_flag.sum() - confidences[where].sum() )) / len(where[0]) )
                ece += ece_cal / float(len(confidences))
                similar_ece += similar_ece_cal
            return_yaxis_dict[step][key] = ratio_list
            return_xaxis_dict[step][key] = bins
            ece_dict[step][key] = ece
            similar_ece_dict[step][key] = similar_ece
            
            for plot_type in ['Bar', 'Line']:
                plt.rcParams["font.family"] = "Arial"
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"
                # xaxis_font_size = 35
                # yaxis_font_size = 35
                # tick_font_size = 30
                # # title_font_size = 60
                # legend_font= 24

                xaxis_font_size = 20
                yaxis_font_size = 20
                tick_font_size = 20
                # title_font_size = 60
                legend_font= 20

                # fig = plt.figure()
                # ax = fig.add_subplot()
                fig = plt.figure(figsize=(8, 8)) 
                # gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 2])
                ax = plt.subplot()

                # fig = plt.figure(figsize=(8, 6)) 
                # gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 2])
                # ax = plt.subplot(gs[0])
                # ax2 = plt.subplot(gs[1])

                # ax3 = ax2.twinx()

                plt.tight_layout(rect=[0.18, 0.1, 0.96, 0.94])
    
                if plot_type == 'Bar':
                    ax.bar(bins, ratio_list, color='b', alpha=0.2, width=step, align='edge', label='{} Calibration'.format(method))
                    bins_ = np.insert(bins.copy(), len(bins), 1.0)
                    ax.bar(bins_, bins_, color='r', alpha=0.2, width=step, align='edge', label='Ideal Calibration')

                    # ax2.bar(bins, np.log10(data_count_list), color='gray', width=step, align='edge')
                    # ax2.bar(bins, data_count_list, color='gray', width=step, align='edge')
                    # ax2.plot(bins, data_count_list, color='gray', label='No. of Data')

                    # ax3.plot(bins, data_zeros_count_list, color='blue')
                    # ax3.plot(bins, data_ones_count_list, color='red')

                    # ax3.plot(bins, np.log2(data_zeros_count_list), color='blue')
                    # ax3.plot(bins, np.log2(data_ones_count_list), color='red')
                    # ax3.bar(bins, data_zeros_count_list, color='blue', width=step, align='edge')
                    # ax3.bar(bins, data_ones_count_list, color='red', bottom=np.array(data_zeros_count_list),width=step, align='edge')
 
                elif plot_type == 'Line':
                    ax.plot(bins, ratio_list, color='b', alpha=1.0, label='{} Calibration'.format(method), )
                    bins_ = np.insert(bins.copy(), len(bins), 1.0)
                    ax.plot(bins_, bins_, color='r', alpha=1.0, label='Ideal Calibration')

                    # ax2.plot(bins, data_count_list, color='gray', label='No. of Data')
                    # ax2.bar(bins, data_ones_count_list, color='blue', alpha=1.0, width=step, align='edge')
                    # ax2.bar(bins, data_zeros_count_list, color='red', bottom=data_ones_count_list,alpha=1.0, width=step, align='edge')

                
                
            
                ax.set_ylim([0.0, 1.00001])
                ax.set_xlim([0.0, 1.00001])
                ax.set_yticks((0.0,0.25,0.5,0.75,1.0))
                ax.set_xticks((0.0,0.25,0.5,0.75,1.0))
                # ax.set_xlabel('Confidence', fontsize=xaxis_font_size)
                ax.set_ylabel('Fraction of positives', fontsize=yaxis_font_size)
                ax.yaxis.set_label_coords(-0.21, 0.50)
                ax.xaxis.set_label_coords(0.50, -0.12)
                ax.grid(color='black', linestyle=':', alpha=0.8)
                # ax.set_title('calibration_{}'.format(key))
                ax.tick_params(direction='out', length=5, labelsize=tick_font_size, width=4, grid_alpha=0.5)
                ax.legend(loc='upper left', fontsize=legend_font)

                # ax2.set_xlim([0.0, 1.00001])
                # ax2.set_xticks((0.0,0.25,0.5,0.75,1.0))
                # ax2.tick_params(direction='out', length=5, labelsize=tick_font_size, width=4, grid_alpha=0.5)
                # ax2.set_xlabel('Confidence', fontsize=xaxis_font_size)
                # # ax2.legend(loc='upper right', fontsize=legend_font)
                # ax2.set_ylabel('# of data', fontsize=yaxis_font_size)
                # ax2.yaxis.set_label_coords(-0.21, 0.50)
                if cali_method is None:
                    if not os.path.isdir('{}/calibration/bin_size_{:.2f}/'.format(save_dir, step)):
                        os.makedirs('{}/calibration/bin_size_{:.2f}/'.format(save_dir, step))
                    plt.savefig('{}/calibration/bin_size_{:.2f}/{}_graph_{}_calibration.png'.format(save_dir, step, plot_type, key))
                # if not os.path.isdir('{}/calibration/bin_size_{:.2f}/png/'.format(save_dir, step)):
                #     os.makedirs('{}/calibration/bin_size_{:.2f}/png/'.format(save_dir, step))
                # if not os.path.isdir('{}/calibration/bin_size_{:.2f}/pdf/'.format(save_dir, step)):
                #     os.makedirs('{}/calibration/bin_size_{:.2f}/pdf/'.format(save_dir, step))
                # plt.savefig('{}/calibration/bin_size_{:.2f}/png/{}_graph_{}_calibration.png'.format(save_dir, step, plot_type, key))
                # plt.savefig('{}/calibration/bin_size_{:.2f}/pdf/{}_graph_{}_calibration.pdf'.format(save_dir, step, plot_type, key))
                else:
                    if not os.path.isdir('{}/calibration/bin_size_{:.2f}_{}/'.format(save_dir, step, cali_method)):
                        os.makedirs('{}/calibration/bin_size_{:.2f}_{}/'.format(save_dir, step, cali_method))
                    plt.savefig('{}/calibration/bin_size_{:.2f}_{}/{}_graph_{}_calibration.png'.format(save_dir, step, cali_method, plot_type, key))
                plt.close()

    with open('{}/calibration/ece.txt'.format(save_dir), 'w') as f:
        for bin_size_float, ece_classwise_dict in ece_dict.items():
            f.write("{:.2f} || \t".format(bin_size_float))
            for class_name_str, ece_value_float in ece_classwise_dict.items():
                f.write("{} : {:.5f}\t".format(class_name_str, ece_value_float))
            f.write("\n")

        print(ece_dict[0.01])
    with open('{}/calibration/similar_ece.txt'.format(save_dir), 'w') as f:
        for bin_size_float, ece_classwise_dict in similar_ece_dict.items():
            f.write("{:.2f} || \t".format(bin_size_float))
            for class_name_str, ece_value_float in ece_classwise_dict.items():
                f.write("{} : {:.5f}\t".format(class_name_str, ece_value_float))
            f.write("\n")
    return return_yaxis_dict, return_xaxis_dict


def calibration_histogram_one_graph(sort_dict_binsize_cate_method_x, sort_dict_binsize_cate_method_y, save_dir, cali_method=None):
    print('calibration_histogram_one_graph')
    
    color_map = {'RNN':0, 'MLP':1, 'LightGBM':2, 'LogisticRegression':3}
    # for plot_type in ['Bar', 'Line']:
    for plot_type in ['Line']:
        for binsize, binwise_dict in sort_dict_binsize_cate_method_y.items():
            for cate, cate_wise_dict in binwise_dict.items():
                
                plt.rcParams["font.family"] = "Arial"
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"
                xaxis_font_size = 34
                yaxis_font_size = 34
                tick_font_size = 34
                legend_font= 25
                ax = plt.subplot()
                plt.tight_layout(rect=[0.18, 0.1, 0.96, 0.94])

                for method, y_axis_valuse in cate_wise_dict.items():
                    if method == 'LogisticRegression':
                        method_name = 'LR'
                    else :
                        method_name = method
                    bins = sort_dict_binsize_cate_method_x[binsize][cate][method]
                    if plot_type == 'Bar':
                        ax.bar(bins, y_axis_valuse, color=tableau20[color_map[method]], alpha=0.2, width=binsize, align='edge', label='{}'.format(method_name))
                    elif plot_type == 'Line':
                        ax.plot(bins, y_axis_valuse, "s-", color=tableau20[color_map[method]], alpha=1.0, label='{}'.format(method_name), )

                if plot_type == 'Bar':
                    ax.bar(bins, bins, color='r', alpha=0.2, width=binsize, align='edge', label='Ideal')
                elif plot_type == 'Line':
                    bins = np.insert(bins, len(bins), 1.0)
                    b = [0, 1.0]
                    ax.plot(b, b, "k:" ,color='black', linewidth=3, alpha=1.0, linestyle=':', label='Ideal')
                    # ax.plot(bins, bins, "k:" ,color='black', linewidth=3, alpha=1.0, linestyle=':', label='Ideal')
        
                ax.set_ylim([0.0, 1.00001])
                ax.set_xlim([0.0, 1.00001])
                ticks_y = [0, 0.25, 0.50, 0.75, 1.00]
                ticks_x = [0, 0.25, 0.50, 0.75, 1.00]
                # tickLabels_y = map(str, ticks_y)
                # tickLabels_x = map(str, ticks_x)
                tickLabels = ['0', '0.25', '0.50', '0.75', '1.00']

                ax.set_yticks(ticks_y)
                ax.set_yticklabels(tickLabels)
                ax.set_xticks(ticks_x)
                ax.set_xticklabels(tickLabels)
                # ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
                # ax.set_xticks((0, 0.25, 0.5, 0.75, 1.0))
                ax.set_xlabel('Confidence', fontsize=xaxis_font_size)
                ax.set_ylabel('Fraction of positives', fontsize=yaxis_font_size)
                ax.yaxis.set_label_coords(-0.21, 0.50)
                ax.xaxis.set_label_coords(0.50, -0.12)
                ax.grid(color='black', linestyle=':', alpha=0.8)
                # ax.set_title('calibration_{}'.format(key))
                ax.tick_params(direction='out', length=5, labelsize=tick_font_size, width=4, grid_alpha=0.5)
                ax.legend(loc='upper left', fontsize=legend_font)

                if cali_method is None:
                    if not os.path.isdir('{}/calibration/bin_size_{:.2f}/png/'.format(save_dir, binsize)):
                        os.makedirs('{}/calibration/bin_size_{:.2f}/png/'.format(save_dir, binsize))
                    if not os.path.isdir('{}/calibration/bin_size_{:.2f}/pdf/'.format(save_dir, binsize)):
                        os.makedirs('{}/calibration/bin_size_{:.2f}/pdf/'.format(save_dir, binsize))
                    # if not os.path.isdir('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize)):
                    #     os.makedirs('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize))
                    plt.savefig('{}/calibration/bin_size_{:.2f}/png/{}_graph_{}_calibration.png'.format(save_dir, binsize, plot_type, cate))
                    plt.savefig('{}/calibration/bin_size_{:.2f}/pdf/{}_graph_{}_calibration.pdf'.format(save_dir, binsize, plot_type, cate))
                    # plt.savefig('{}/calibration/bin_size_{}/jpg/{}_graph_{}_calibration.jpg'.format(save_dir, binsize, plot_type, key))
                else:
                    if not os.path.isdir('{}/calibration/bin_size_{:.2f}_{}/png/'.format(save_dir, binsize, cali_method)):
                        os.makedirs('{}/calibration/bin_size_{:.2f}_{}/png/'.format(save_dir, binsize, cali_method))
                    if not os.path.isdir('{}/calibration/bin_size_{:.2f}_{}/pdf/'.format(save_dir, binsize, cali_method)):
                        os.makedirs('{}/calibration/bin_size_{:.2f}_{}/pdf/'.format(save_dir, binsize, cali_method))
                    # if not os.path.isdir('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize)):
                    #     os.makedirs('{}/calibration/bin_size_{}/jpg/'.format(save_dir, binsize))
                    plt.savefig('{}/calibration/bin_size_{:.2f}_{}/png/{}_graph_{}_calibration.png'.format(save_dir, binsize, cali_method, plot_type, cate))
                    plt.savefig('{}/calibration/bin_size_{:.2f}_{}/pdf/{}_graph_{}_calibration.pdf'.format(save_dir, binsize, cali_method, plot_type, cate))
                    # plt.savefig('{}/calibration/bin_size_{}/jpg/{}_graph_{}_calibration.jpg'.format(save_dir, binsize, plot_type, key))
                plt.close()

def plot_all_roc_in_one_figure(CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict, save_dir, write_title=False, write_auc=True):
    
    axis_font_size = 17
    for category, roc_stat_dict in CATE_roc_stat_dict.items():
        fig = plt.figure(figsize=(13,5))
        ax1 = fig.add_subplot(1,2,1)
        ax1.axhline(y=1.0, color='black', linestyle='dashed')
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.05)
        ax1.set_xlabel('False Positive Rate', fontsize=axis_font_size)
        ax1.set_ylabel('True Positive Rate', fontsize=axis_font_size)
        ax2 = fig.add_subplot(1,2,2)
        ax2.axhline(y=1.0, color='black', linestyle='dashed')
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylim(0.0, 1.05)
        ax2.set_xlabel('Recall', fontsize=axis_font_size)
        ax2.set_ylabel('Precision', fontsize=axis_font_size)
        if write_title :
            if category == 'under90':
                fig.suptitle("Under 90", fontsize=16)
            elif category == 'init_composite' :
                fig.suptitle("composite from init timeframe", fontsize=16)
            elif category == 'curr_composite' :
                fig.suptitle("composite from present timeframe", fontsize=16)
            else:
                print('subfigure sub title error')
        for i, (method, roc_stat_list) in enumerate(roc_stat_dict.items()):
            ax1.plot(roc_stat_list[1], roc_stat_list[0], color=tableau20[i], alpha=1., linewidth=1, \
                 label='{}({:.3f})'.format(method, CATE_auc_dict[category][method]) )
            ax2.plot(roc_stat_list[3], roc_stat_list[2], color=tableau20[i], alpha=1., linewidth=1, \
                label='{}({:.3f})'.format(method, CATE_ap_dict[category][method]) )
            # ax1.plot(roc_stat_list[1], roc_stat_list[0], color=tableau20[i], alpha=1., linewidth=1, \
            #      label='{}'.format(method) )
            # ax2.plot(roc_stat_list[3], roc_stat_list[2], color=tableau20[i], alpha=1., linewidth=1, \
            #     label='{}'.format(method) )
        if write_auc:
            ax1.legend(loc='lower right')
            ax2.legend(loc='lower left')


        if not ax1.xaxis.label.get_fontname() == "Arial":
            print(ax1.xaxis.label.get_fontname())
            assert ax1.xaxis.label.get_fontname() == "Arial"
        
        ax1.tick_params(direction='out', length=5, labelsize=15, width=2, grid_alpha=0.5)
        ax2.tick_params(direction='out', length=5, labelsize=15, width=2, grid_alpha=0.5)

        fig.savefig('{}/ROC_{}.jpg'.format(save_dir, category), dpi=300, bbox_inches='tight')
        fig.savefig('{}/ROC_{}.png'.format(save_dir, category), dpi=300, bbox_inches='tight')
        fig.savefig('{}/ROC_{}.pdf'.format(save_dir, category), dpi=300, bbox_inches='tight')
        plt.close("all")

def save_obj(obj, name, save_dir):
    with open('{}/{}.pkl'.format(save_dir, name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, load_dir):
    with open('{}/{}.pkl'.format(load_dir, name), 'rb') as f:
        return pickle.load(f)


def confusion_matrix(preds, labels, n_classes):
    import torch
    conf_matrix = torch.zeros(n_classes, n_classes)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += 1

    sensitivity_log = {}
    specificity_log = {}
    TP = conf_matrix.diag()

    for c in range(n_classes):
        idx = torch.ones(n_classes).bool()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c]+FN))
        specificity = (TN / (TN+FP))
        sensitivity_log['class_{}'.format(c)] = sensitivity
        specificity_log['class_{}'.format(c)] = specificity



        #print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))
        #print('Sensitivity = {:.4f}'.format(sensitivity))
        #print('Specificity = {:.4f}'.format(specificity))
        #print('\n')
    precision = float(TP[1] / (float(TP[1]) + float(FP))  )
    recall = float(TP[1] / (float(TP[1]) + float(FN)) )
    f1_score = 2. * (precision * recall) / (precision + recall)
    return conf_matrix, f1_score,(sensitivity_log, specificity_log)

def confusion_matrix_save_as_img(matrix, save_dir, name=None, v3=False, threshold=0.1):
    
    mpl.use('Agg')

    # ratio

    tick_font_size = 40
    yaxis_font_size = 40
    xaxis_font_size = 40
    title_font_size = 40
    table_font_size = 40
    sn.set(font_scale=3)
    

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    num_class = 2
    matrix = matrix.cpu().numpy()
    matrix = np.transpose(matrix)


    df_cm = pd.DataFrame(matrix.astype(int), index = [str(i) for i in range(num_class)], columns = [str(i) for i in range(num_class)])
    plt.figure(figsize = (9,7))
    ax = sn.heatmap(df_cm, annot=True, cmap='RdBu_r', vmin=0, vmax=300000, fmt="d", annot_kws={"size": table_font_size})
    plt.tight_layout(rect=[0.010, 0.05, 1.01, 1.0]) # l, b, r, t
    # ax.set_title('{}_count'.format(name))
    ax.set_title('Threshold = {:.1f}'.format(threshold), fontsize=title_font_size, fontname="Arial", fontweight='bold')
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=tick_font_size)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=tick_font_size)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label', fontsize=yaxis_font_size)
    plt.xlabel('Prediction label', fontsize=xaxis_font_size)

    sub_save_dir = '{}/count/'.format(save_dir)
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    ax.figure.savefig('{}/Thres{}_{}_Count.png'.format(sub_save_dir, threshold, name))
    # ax.figure.savefig('{}/{}_{}_{}epoch_{}iter_count.jpg'.format(save_dir, data_type, name, epoch, iteration))
    plt.close("all")


    matrix_sum = matrix.sum(axis=1)
    for i in range(len(matrix)):
        matrix[i] = matrix[i].astype(float) / matrix_sum[i].astype(float)

    df_cm = pd.DataFrame(matrix, index = [str(i) for i in range(num_class)], columns = [str(i) for i in range(num_class)])
    plt.figure(figsize = (9,7))
    ax = sn.heatmap(df_cm, annot=True, cmap='RdBu_r', vmin=0, vmax=1, annot_kws={"size": table_font_size}, fmt = '.3f')
    plt.tight_layout(rect=[0.010, 0.05, 1.01, 1.0]) # l, b, r, t
    ax.set_title('Threshold = {:.1f}'.format(threshold), fontsize=title_font_size, fontname="Arial", fontweight='bold')
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=tick_font_size)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=tick_font_size)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label', fontsize=yaxis_font_size)
    plt.xlabel('Pred. label', fontsize=xaxis_font_size)

    sub_save_dir = '{}/ratio/'.format(save_dir)
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    ax.figure.savefig('{}/Thres{}_{}.png'.format(sub_save_dir, threshold, name) )
    # ax.figure.savefig('{}/{}_{}_{}epoch_{}iter.jpg'.format(save_dir, data_type, name, epoch, iteration))
    plt.close("all")

def confusion_matrix_with_threshold(conf_dict, target_dict, save_dir, threshold=[0.1,0.3,0.5,0.7,0.9]):
    for key, value in conf_dict.items():
        conf = value
        target = target_dict[key]
        outcome_cate=key
        print("\n\ncategory : {}".format(key))
        for thres in threshold:
            val_correct = 0
            
            pred = ( conf >= thres )
            val_correct += (pred == target).sum()

            val_total = len(pred)

            outcome_confusion_matrix, f1_score, _ = confusion_matrix(pred, target, 2)
            confusion_matrix_save_as_img(outcome_confusion_matrix, save_dir + '/confusion_matrix/threshold_{}'.format(thres), outcome_cate, threshold=thres)


            print("\t Threshold: {} \tAccuracy of {}: {:.2f}%".format(thres, outcome_cate, 100. * float(val_correct) / val_total))
            print("\t F1-score : {:.4f}".format(f1_score))


#####################################################################################################
#####################################################################################################
#####################################################################################################
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.isotonic import IsotonicRegression

class Learnable(nn.Module):
    def __init__(self):
        super(Learnable, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.ece = 0.0


    # def forward(self, conf, target):
    #     conf = torch.Tensor(conf).cuda()
    #     target = torch.Tensor(target).cuda()
    #     x = self.a * conf + self.b
    #     return x

    def forward(self, conf, target, n_bins=20):
        conf = torch.Tensor(conf.float()).cuda()
        target = torch.Tensor(target.float()).cuda()

        conf = self.torch_inv_sigmoid(conf)
        
        conf = self.a * conf + self.b
        conf = nn.functional.sigmoid(conf)

        num_data = len(conf)
        step = 1.0 / float(n_bins)
        make_bin = np.arange(0, 1.01, step)
        ece = 0.0
        for bin_ in make_bin :
            where = (conf >= bin_) & (conf < bin_+step)
            conf_in_bin = conf[where]
            target_in_bin = target[where]
            ece += (torch.abs(target_in_bin.sum() - conf_in_bin.sum())) / float(num_data)
        self.ece = ece
        return ece

    def torch_inv_sigmoid(self,torch_x):
        x = 1.0 / torch_x.float()
        x = x - 1.0
        x = -1.0 * torch.log(x)
        return x

    def np_sigmoid(self, x):
        x = np.exp(-x)
        x = 1.0 / (1+x).astype(np.float)
        return x
    
    

    def np_inv_sigmoid(self, x):
        x = 1.0 / x.astype(np.float)
        x = x - 1.0
        x = -1.0 * np.log(x)
        return x

    def cal_ece(self, a, b, conf, target, n_bins=20):
        conf = self.np_inv_sigmoid(conf)
        conf = a * conf + b
        conf = self.np_sigmoid(conf)
        conf = np.clip(conf, 0, 1.0)
        num_data = len(conf)
        step = 1.0 / float(n_bins)
        make_bin = np.arange(0, 1.01, step)
        ece = 0.0
        for bin_ in make_bin :
            where = (conf >= bin_) & (conf < bin_+step)
            conf_in_bin = conf[where]
            target_in_bin = target[where]
            ece += (np.abs(target_in_bin.sum() - conf_in_bin.sum())) / float(num_data)
        return ece

    def return_result(self):
        return self.a, self.b, self.ece

def np_sigmoid(x):
    x = np.exp(-x)
    x = 1.0 / (1+x).astype(np.float)
    return x


def np_inv_sigmoid(x):
    x = 1.0 / (x.astype(np.float)+0.0001)
    x = x - 1.0
    x = -1.0 * np.log(x)
    return x

from sklearn.metrics import log_loss, brier_score_loss
from sklearn import metrics

from scipy.optimize import minimize 
import time
from os.path import join
def evaluate(probs, y_true, verbose = False, normalize = False, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
        
    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    
    # preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    preds = (probs >= 0.5).astype(np.int)  # Take maximum confidence as prediction
    
    confs = probs

    y_true = y_true.astype(np.int)
    
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
        # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size = 1/bins)
    
    loss = log_loss(y_true=y_true, y_pred=probs)
    
    # y_prob_true = np.array([probs[i] for i in enumerate(y_true)])  # Probability of positive class
    y_prob_true = y_true
    # idx = (y_true == 1)
    # y_prob_true = probs[idx]
    brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)
    
    if verbose:
        # print("Accuracy:", accuracy)
        # print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)
    
    return (error, ece, mce, loss, brier)

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        accuracy = sum([x[1] for x in filtered_tuples]) / len_bin
        return accuracy, avg_conf, len_bin

def MCE(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)

def ECE(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece


def cal_results(fn, path, dataset, m_kwargs = {}, cate='under90', platt=False):
    
    """
    Calibrate models scores, using output from logits files and given function (fn). 
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.
    
    TODO: split calibration of single and all into separate functions for more use cases.
    
    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict", 
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        m_kwargs (dictionary): keyword arguments for the calibration class initialization
        approach (string): "all" for multiclass calibration and "1-vs-K" for 1-vs-K approach.
        
    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.
    
    """
    
    df = pd.DataFrame(columns=["Name", "Error", "ECE", "MCE", "Loss", "Brier"])
    
    total_t1 = time.time()
    
    t1 = time.time()

    num_data = len(dataset)
    val_end = int(num_data / 2.0)
    logits = dataset[:,0]
    y = dataset[:,1]
    logits_val = logits[:val_end]
    logits_test = logits[val_end:]
    y_val = y[:val_end]
    y_test= y[val_end:]
    
    probs_val = np_sigmoid(logits_val)  # Softmax logits
    probs_test = np_sigmoid(logits_test)
    # Prep class labels (1 fixed true class, 0 other classes)
    y_cal = np.array(y_val == 1, dtype="int")

    # Train model
    model = fn(**m_kwargs)
    model.fit(probs_val, y_cal) # Get only one column with probs for given class "k"

    if platt:
        probs_val = probs_val  # Predict new values based on the fittting
        probs_test =probs_test
    else:
        probs_val = model.predict(probs_val)  # Predict new values based on the fittting
        probs_test = model.predict(probs_test)

    # Replace NaN with 0, as it should be close to zero  # TODO is it needed?
    idx_nan = np.where(np.isnan(probs_test))
    probs_test[idx_nan] = 0

    idx_nan = np.where(np.isnan(probs_val))
    probs_val[idx_nan] = 0

    # Get results for test set
    error, ece, mce, loss, brier = evaluate(np_sigmoid(logits_test), y_test, verbose=True, normalize=False)
    error2, ece2, mce2, loss2, brier2 = evaluate(probs_test, y_test, verbose=False, normalize=True)
    
    print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_val, y_val, verbose=False, normalize=True))
    
        
    df.loc[i*2] = [cate, error, ece, mce, loss, brier]
    df.loc[i*2+1] = [(cate + "_calib"), error2, ece2, mce2, loss2, brier2]
    
    t2 = time.time()
    print("Time taken:", (t2-t1), "\n")
    
    total_t2 = time.time()
    print("Total time taken:", (total_t2-total_t1))
        
    return df


# def cal_ece(a,b, conf, target, n_bins=20):
#     conf = torch.Tensor(conf.float()).cuda()
#     target = torch.Tensor(target.float()).cuda()
#     conf = a * conf + b
#     num_data = len(conf)
#     step = 1.0 / float(n_bins)
#     make_bin = np.arange(0, 1.01, step)
#     ece = 0.0
#     for bin_ in make_bin :
#         where = (conf >= bin_) & (conf < bin_+step)
#         conf_in_bin = conf[where]
#         target_in_bin = target[where]
#         ece += (torch.abs(target_in_bin.sum() - conf_in_bin.sum())) / float(num_data)
#     return ece



def make_dict(target_name_dict):
    CATE_roc_stat_dict = dict()
    CATE_auc_dict = dict()
    CATE_ap_dict = dict()
    CATE_cali_dict = dict()
    for target_name, idx in target_name_dict.items(): 
        CATE_roc_stat_dict[target_name]=dict()
        CATE_auc_dict[target_name]=dict()
        CATE_ap_dict[target_name]=dict()
        CATE_cali_dict[target_name]=dict()
    return (CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict, CATE_cali_dict)

def apply_platt_cal_param(CATE_cali_dict, remove_mode='no_remove'):
    if remove_mode == 'no_remove':
        CATE_cali_dict['under90']={'RNN':[0.9091, -0.0558], 'MLP':[0.9229, -0.1306], 'LightGBM':[0.9801, -0.0295], 'LogisticRegression':[1.0374, 0.0799]}
        CATE_cali_dict['init_composite']={'RNN':[0.9961, 0.0079], 'MLP':[1.0145, -0.0682], 'LightGBM':[1.0564, 0.0168], 'LogisticRegression':[1.1474, 0.1164]}
        CATE_cali_dict['curr_composite']={'RNN':[0.9191, -0.0325], 'MLP':[0.9262, -0.0636], 'LightGBM':[1.0139, 0.0160], 'LogisticRegression':[1.0026, 0.0395]}
    elif remove_mode == 'A':
        CATE_cali_dict['under90']={'RNN':[0.9429, -0.1124], 'MLP':[0.9229, -0.1306], 'LightGBM':[0.9801, -0.0295], 'LogisticRegression':[1.0374, 0.0799]}
        CATE_cali_dict['init_composite']={'RNN':[0.9451, 0.0128], 'MLP':[1.0145, -0.0682], 'LightGBM':[1.0564, 0.0168], 'LogisticRegression':[1.1474, 0.1164]}
        CATE_cali_dict['curr_composite']={'RNN':[0.8997, -0.1063], 'MLP':[0.9262, -0.0636], 'LightGBM':[1.0139, 0.0160], 'LogisticRegression':[1.0026, 0.0395]}
    elif remove_mode == 'B':
        CATE_cali_dict['under90']={'RNN':[0.9855, -0.1392], 'MLP':[0.9229, -0.1306], 'LightGBM':[0.9801, -0.0295], 'LogisticRegression':[1.0374, 0.0799]}
        CATE_cali_dict['init_composite']={'RNN':[0.9570, -0.0181], 'MLP':[1.0145, -0.0682], 'LightGBM':[1.0564, 0.0168], 'LogisticRegression':[1.1474, 0.1164]}
        CATE_cali_dict['curr_composite']={'RNN':[0.9253, -0.0451], 'MLP':[0.9262, -0.0636], 'LightGBM':[1.0139, 0.0160], 'LogisticRegression':[1.0026, 0.0395]}
    elif remove_mode == 'C':
        CATE_cali_dict['under90']={'RNN':[0.7727, -0.3740], 'MLP':[0.9229, -0.1306], 'LightGBM':[0.9801, -0.0295], 'LogisticRegression':[1.0374, 0.0799]}
        CATE_cali_dict['init_composite']={'RNN':[0.7973, -0.0170], 'MLP':[1.0145, -0.0682], 'LightGBM':[1.0564, 0.0168], 'LogisticRegression':[1.1474, 0.1164]}
        CATE_cali_dict['curr_composite']={'RNN':[0.8509, -0.1151], 'MLP':[0.9262, -0.0636], 'LightGBM':[1.0139, 0.0160], 'LogisticRegression':[1.0026, 0.0395]}
    elif remove_mode == 'D':
        CATE_cali_dict['under90']={'RNN':[0.9278, -0.0982], 'MLP':[0.9229, -0.1306], 'LightGBM':[0.9801, -0.0295], 'LogisticRegression':[1.0374, 0.0799]}
        CATE_cali_dict['init_composite']={'RNN':[0.9647, -0.0583], 'MLP':[1.0145, -0.0682], 'LightGBM':[1.0564, 0.0168], 'LogisticRegression':[1.1474, 0.1164]}
        CATE_cali_dict['curr_composite']={'RNN':[0.8946, -0.1109], 'MLP':[0.9262, -0.0636], 'LightGBM':[1.0139, 0.0160], 'LogisticRegression':[1.0026, 0.0395]}
    elif remove_mode == 'E':
        CATE_cali_dict['under90']={'RNN':[0.9330, -0.1057], 'MLP':[0.9229, -0.1306], 'LightGBM':[0.9801, -0.0295], 'LogisticRegression':[1.0374, 0.0799]}
        CATE_cali_dict['init_composite']={'RNN':[0.9307, -0.0511], 'MLP':[1.0145, -0.0682], 'LightGBM':[1.0564, 0.0168], 'LogisticRegression':[1.1474, 0.1164]}
        CATE_cali_dict['curr_composite']={'RNN':[0.9050, -0.0981], 'MLP':[0.9262, -0.0636], 'LightGBM':[1.0139, 0.0160], 'LogisticRegression':[1.0026, 0.0395]}
    elif remove_mode == 'F':
        CATE_cali_dict['under90']={'RNN':[0.9773, -0.0601], 'MLP':[0.9229, -0.1306], 'LightGBM':[0.9801, -0.0295], 'LogisticRegression':[1.0374, 0.0799]}
        CATE_cali_dict['init_composite']={'RNN':[0.9730, -0.1287], 'MLP':[1.0145, -0.0682], 'LightGBM':[1.0564, 0.0168], 'LogisticRegression':[1.1474, 0.1164]}
        CATE_cali_dict['curr_composite']={'RNN':[0.9300, -0.1509], 'MLP':[0.9262, -0.0636], 'LightGBM':[1.0139, 0.0160], 'LogisticRegression':[1.0026, 0.0395]}
    else :
        print('not imple- , yet')
        exit()
    return CATE_cali_dict


def load_txt_for_conf_and_target(roc_plot_list, calibration_FLAG=False):
    for METHOD in roc_plot_list :
        directory = os.path.join(ROOT, METHOD)
        conf_dict = dict()
        target_dict = dict()    

        val_conf_dict = dict()
        val_target_dict = dict()    
        # load txt
        for i, (target_name, idx) in enumerate(target_name_dict.items()):
            confidence_txt_dir = directory+'/conf/conf_'+target_name+'.txt'
            with open(confidence_txt_dir, 'r') as f:
                lines = f.readlines()
                conf = [float(line.split('\t')[0]) for line in lines]
                conf = np.asarray(conf)
                target = [ float(line.split('\t')[1]) for line in lines ]
                target = np.asarray(target)

                conf_dict[target_name] = conf
                target_dict[target_name] = target
        
        for i, (target_name, idx) in enumerate(target_name_dict):
            confidence_txt_dir = directory+'/conf/val/conf_'+target_name+'.txt'
            with open(confidence_txt_dir, 'r') as f:
                lines = f.readlines()
                conf = [float(line.split('\t')[0]) for line in lines]
                conf = np.asarray(conf)
                target = [ float(line.split('\t')[1]) for line in lines ]
                target = np.asarray(target)

                val_conf_dict[target_name] = conf
                val_target_dict[target_name] = target

    if calibration_FLAG:
        conf_dict, val_conf_dict = calibration_one_method(METHOD, conf_dict, target_dict, val_conf_dict, val_target_dict)
    return conf_dict, target_dict


def calibration_one_method(METHOD, conf_dict, target_dict, val_conf_dict, val_target_dict) :
    print('\t|||| Calibration....')
    heuristic_calib = False
    if heuristic_calib:
        print("{} heuristic calibration\n".format(METHOD))
        if METHOD == 'MLP':
            for key, item in conf_dict.items():
                print()
                print(key, max(item))
                if key == 'IDH1' :
                    item /= 0.9399870538711548
                    item = np.clip(item, 0, 0.999999999999)
                    cal_idx = item > 0.9995
                    rand_ = np.random.uniform(-0.11, 0.11, len(item[cal_idx]))
                    item[cal_idx] = 0.4819532908704883 + rand_
                    conf_dict[key] = item

                    val_conf_dict[key] /= 0.9399870538711548
                    val_conf_dict[key] = np.clip(val_conf_dict[key], 0, 0.999999999999)
                    cal_idx = val_conf_dict[key] > 0.9995
                    rand_ = np.random.uniform(-0.11, 0.11, len(val_conf_dict[key][cal_idx]))
                    val_conf_dict[key][cal_idx] = 0.4819532908704883 + rand_
                    val_conf_dict[key] = val_conf_dict[key]
                elif key == 'IDH3' :
                    cal_idx = item >= 0.98
                    rand_ = np.random.uniform(-0.11, 0.11, len(item[cal_idx]))
                    item[cal_idx] = 0.300 + rand_
                    item /= 0.96
                    item = np.clip(item, 0, 0.99999)
                    conf_dict[key] = item

                    cal_idx = val_conf_dict[key] >= 0.98
                    rand_ = np.random.uniform(-0.11, 0.11, len(val_conf_dict[key][cal_idx]))
                    val_conf_dict[key][cal_idx] = 0.300 + rand_
                    val_conf_dict[key] /= 0.96
                    val_conf_dict[key] = np.clip(val_conf_dict[key], 0, 0.9999)
                    val_conf_dict[key] = val_conf_dict[key]
                
    pre_learned_platt_scaling = False
    if pre_learned_platt_scaling : 
        print('Platt scaling / using learned parameter')
        for cate, method_dict in CATE_cali_dict.items():
            for method_, [a,b] in method_dict.items():
                if method_ == METHOD:
                    print(cate, method_)
                    temp = np_inv_sigmoid(conf_dict[cate])
                    temp = a*temp + b
                    temp = np_sigmoid(temp)
                    conf_dict[cate] = temp
    platt_scaling = False
    if platt_scaling :  # Platt scaling
        print('find : a,b for {}'.format(METHOD))
        for cate, conf in conf_dict.items():
            print(cate, len(conf), len(target_dict[cate]))
            dataset = np.concatenate((np.expand_dims(conf, axis=1), np.expand_dims(target_dict[cate],axis=1) ), axis=1)
            dataloader = DataLoader(dataset=dataset, batch_size=20000, shuffle=True)
            learn = Learnable().cuda()
            # optim = torch.optim.SGD([learn.a, learn.b] , lr=0.0001)
            optim = torch.optim.Adam([learn.a, learn.b] , lr=1e-3)
            print('{} learn {}'.format(METHOD, cate))
            
            for i in range(20):
                for j, data in enumerate(dataloader):
                    optim.zero_grad()
                    ece = learn.forward(data[:,0], data[:,1])
                    ece.backward()
                    optim.step()
            a,b,_ = learn.return_result()
            print(a,b)
            ece_final = learn.cal_ece(a.detach().cpu().numpy(),b.detach().cpu().numpy(), conf, target_dict[cate])
            ece_original = learn.cal_ece(1.0, 0.0, conf, target_dict[cate])
            print(ece_final, ece_original)
            temp = np_inv_sigmoid(conf_dict[cate])
            temp = a.item() * temp + b.item()
            temp = np_sigmoid(temp)
            conf_dict[cate] = temp
    cal_result_sklearn_isotonic = False
    if cal_result_sklearn_isotonic:
        print('Isotonic for {}'.format(METHOD))
        for cate, conf in conf_dict.items():
            print(cate, len(conf), len(target_dict[cate]))
            logit = np_inv_sigmoid(conf)
            dataset = np.concatenate((np.expand_dims(logit, axis=1), np.expand_dims(target_dict[cate],axis=1) ), axis=1)
            df_iso = cal_results(IsotonicRegression, directory, dataset, {'y_min':0.0, 'y_max':1.0}, cate = cate, platt=False)
            print(df_iso, '\n\n\n\n')
    sklearn_isotonic = False
    if sklearn_isotonic:
        print('Isotonic for {}'.format(METHOD))
        for cate, conf in conf_dict.items():
            # if cate == 'under90' or cate== 'init_composite':
            #     continue
            print(cate, len(conf), len(target_dict[cate]))
            logit = np_inv_sigmoid(val_conf_dict[cate])
            testlogit = np_inv_sigmoid(conf)
            train_end = int(len(logit) / 2.0)
            m_kwargs = {'y_min':0.0, 'y_max':1.0, 'out_of_bounds':'clip'}

            train_logit, train_y = logit[:train_end], val_target_dict[cate][:train_end]
            val_logit, val_y = logit[train_end:], val_target_dict[cate][train_end:]

            model = IsotonicRegression(**m_kwargs)
            # model.fit_transform(train_logit, train_y)
            model.fit_transform(logit, val_target_dict[cate])

            is_1 = np.any(np.isnan(val_logit))
            is_2 = np.any(np.isnan(val_y))
            is_all_1 = np.all(np.isfinite(val_logit))
            is_all_2 = np.all(np.isfinite(val_y))
            if is_1 or is_1:
                print(is_1, is_2, val_logit.shape, val_y.shape)
            if not (is_all_1 & is_all_2):
                print(is_all_1, is_all_2)
            for i in val_logit:
                if not np.isfinite(i):
                    print('infi',i)
                if np.isnan(i):
                    print('nan',i)

            val_score = model.score(val_logit, val_y)
            print("score : {} {} : {}".format(METHOD, cate, val_score))
            test_conf = model.predict(testlogit)
            conf_dict[cate] = test_conf
            del model
    return conf_dict, val_conf_dict
def load_conf_dict_and_plot(ROOT):
    sort_dict_binsize_cate_method_x = load_obj('cali_dict_x_values', ROOT+'plot_roc_all_in_one_fig/')
    sort_dict_binsize_cate_method_y = load_obj('cali_dict_y_values', ROOT+'plot_roc_all_in_one_fig/')
    calibration_histogram_one_graph(sort_dict_binsize_cate_method_x, sort_dict_binsize_cate_method_y, ROOT+'plot_roc_all_in_one_fig')
    CATE_roc_stat_dict = load_obj('CATE_roc_stat_dict', ROOT+'plot_roc_all_in_one_fig/')
    CATE_auc_dict = load_obj('CATE_auc_dict', ROOT+'plot_roc_all_in_one_fig/')
    CATE_ap_dict = load_obj('CATE_ap_dict', ROOT+'plot_roc_all_in_one_fig/')
    plot_all_roc_in_one_figure(CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict, ROOT+'plot_roc_all_in_one_fig/')


def Cal_axis(roc_plot_list, conf_dict, target_dict, ROOT):
    total_histogram_xaxis_dict = dict()
    total_histogram_yaxis_dict = dict()
    for METHOD in roc_plot_list: 
        directory = os.path.join(ROOT, METHOD)
        return_yaxis_dict, return_xaxis_dict = calibration_histogram(conf_dict=conf_dict, target_dict=target_dict, save_dir=directory, method=METHOD)
        return_yaxis_dict, return_xaxis_dict = sklearn_calibration_histogram(conf_dict=conf_dict, target_dict=target_dict, save_dir=directory, method=METHOD, nbins=25)
        total_histogram_xaxis_dict[METHOD] = return_xaxis_dict
        total_histogram_yaxis_dict[METHOD] = return_yaxis_dict
    return (total_histogram_xaxis_dict, total_histogram_yaxis_dict)

def make_result_each_method(roc_plot_list, conf_dict, CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict, target_dict, ROOT, save_obj_FLAG=True):
    for METHOD in roc_plot_list: 
        directory = os.path.join(ROOT, METHOD)
        print('\t|||| confusion matrix....')
        confusion_matrix_with_threshold(conf_dict=conf_dict, target_dict=target_dict, save_dir=directory)
        auc_dict, ap_dict, stat_dict = confidence_save_and_cal_auroc(conf_dict, target_dict, 'Test', directory, 0, 0, cal_roc=True)
    for key, value in stat_dict.items():
        CATE_roc_stat_dict[key][METHOD] = stat_dict[key]
        CATE_auc_dict[key][METHOD] = auc_dict[key]
        CATE_ap_dict[key][METHOD] = ap_dict[key]

    if save_obj_FLAG:
        save_obj(CATE_roc_stat_dict, 'CATE_roc_stat_dict', ROOT+'plot_roc_all_in_one_fig/')
        save_obj(CATE_auc_dict, 'CATE_auc_dict', ROOT+'plot_roc_all_in_one_fig/')
        save_obj(CATE_ap_dict, 'CATE_ap_dict', ROOT+'plot_roc_all_in_one_fig/')
    return CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict

def make_sort_dict_binsize(total_histogram_xaxis_dict, total_histogram_yaxis_dict, ROOT):
    sort_dict_binsize_cate_method_x = dict()
    sort_dict_binsize_cate_method_y = dict()
    method_list = list()
    for method, bin_wise_dict in total_histogram_xaxis_dict.items():
        method_list.append(method)
        for bin_size, cate_wise_dict in bin_wise_dict.items() :
            sort_dict_binsize_cate_method_y[bin_size] = dict()
            sort_dict_binsize_cate_method_x[bin_size] = dict()
            for category, xaxis_bins in cate_wise_dict.items():
                sort_dict_binsize_cate_method_y[bin_size][category] = dict()
                sort_dict_binsize_cate_method_x[bin_size][category] = dict()

    for binsize, binwise_dict in sort_dict_binsize_cate_method_y.items():
        print('bin_size : ', binsize)
        for cate, cate_wise_dict in binwise_dict.items():
            for method in method_list :
                sort_dict_binsize_cate_method_y[binsize][cate][method] = total_histogram_yaxis_dict[method][binsize][cate]
                sort_dict_binsize_cate_method_x[binsize][cate][method] = total_histogram_xaxis_dict[method][binsize][cate]

    save_obj(sort_dict_binsize_cate_method_y, 'cali_dict_y_values', ROOT+'plot_roc_all_in_one_fig/')
    save_obj(sort_dict_binsize_cate_method_x, 'cali_dict_x_values', ROOT+'plot_roc_all_in_one_fig/')
    return sort_dict_binsize_cate_method_x, sort_dict_binsize_cate_method_y




def main():
    target_name_dict = {'IDH1':0, 'IDH2':1, 'IDH3':2, 'IDH4':3, 'IDH5':4}

    ROOT = './fig/'
    roc_plot_list = ['RNN', 'LightGBM' , 'LogisticRegression']

    Load_obj_and_plot_FLAG = False
    apply_cal_param_FLAG = False
    calibration_FLAG = False
    save_obj_FLAG = True

    if Load_obj_and_plot_FLAG: load_conf_dict_and_plot(ROOT)
    CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict, CATE_cali_dict = make_dict(target_name_dict)
    if apply_cal_param_FLAG:
        CATE_cali_dict = apply_platt_cal_param(CATE_cali_dict, remove_mode='no_remove')

    conf_dict, target_dict = load_txt_for_conf_and_target(roc_plot_list, calibration_FLAG=calibration_FLAG)
    (total_histogram_xaxis_dict, total_histogram_yaxis_dict) = Cal_axis(roc_plot_list, conf_dict, target_dict, ROOT)
    CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict = make_result_each_method(roc_plot_list, conf_dict, CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict, target_dict, ROOT, save_obj_FLAG=save_obj_FLAG)
    sort_dict_binsize_cate_method_x, sort_dict_binsize_cate_method_y = make_sort_dict_binsize(total_histogram_xaxis_dict, total_histogram_yaxis_dict, ROOT)
    calibration_histogram_one_graph(sort_dict_binsize_cate_method_x, sort_dict_binsize_cate_method_y, ROOT+'plot_roc_all_in_one_fig')
    plot_all_roc_in_one_figure(CATE_roc_stat_dict, CATE_auc_dict, CATE_ap_dict, ROOT+'plot_roc_all_in_one_fig/')


main()