def confusion_matrix_save_as_img(matrix, save_dir, epoch=0, name=None):
    import seaborn as sn
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt


    save_dir = save_dir+'confusion_matrix/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if name == 'sbp':
        num_class = 7
    else :
        num_class = 5
    matrix = np.transpose(matrix)
    matrix_sum = matrix.sum(axis=1)
    for i in range(len(matrix)):
        matrix[i] /= matrix_sum[i]

    df_cm = pd.DataFrame(matrix, index = [str(i) for i in range(num_class)], columns = [str(i) for i in range(num_class)])
    plt.figure(figsize = (7,5))
    ax = sn.heatmap(df_cm, annot=True, cmap='RdBu_r', vmin=0, vmax=1)
    ax.set_title('{}_{}epoch'.format(name, epoch))
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('true label')
    plt.xlabel('pred label')

    ax.figure.savefig('{}/{}_{}epoch.jpg'.format(save_dir, name, epoch))
    plt.close("all")
