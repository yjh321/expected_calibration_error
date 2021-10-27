import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metric


def Expected_Calibration_Error(confidence, true_label, pred_label, bins, n_classes, save_path):
    x = []
    for i in range(bins):
        x_point = 1/n_classes + i*(1-(1/n_classes))/bins
        x.append(x_point)
    x_ = x.copy()
    x_.append(1.0)

    ratio_arr = []
    acc_arr = []
    ece = 0
    for i in range(bins):
        conf_bin = confidence[confidence>=x[i]]
        conf_bin = conf_bin[conf_bin<x_[i+1]]
        index = np.where(np.isin(confidence, conf_bin)==True)
        true_bin = true_label[index]
        pred_bin = pred_label[index]
        ratio_bin = len(true_bin)/len(true_label)
        if len(true_bin)>0:
            acc_bin = metric.accuracy_score(true_bin, pred_bin)
            ece_bin = np.abs(acc_bin - conf_bin.mean())*ratio_bin
            ece += ece_bin
            ratio_arr.append(ratio_bin)
            acc_arr.append(acc_bin)
        else:
            acc_bin = 0
            ratio_arr.append(0)
            acc_arr.append(0)
        
    plt.clf()
    plt.ylim(0, 1)
    plt.bar(x, ratio_arr, width=0.05, align='edge', color='b')
    plt.xlabel('confidence')
    plt.ylabel('% samples')
    plt.savefig('{}/samples.png'.format(save_path))

    plt.clf()
    plt.ylim(0, 1)
    plt.bar(x, acc_arr, width=0.05, align='edge', color='b')
    plt.plot(x_, x_, color='r', linestyle='--')
    plt.xlabel('confidence')
    plt.ylabel('Accuracy')
    plt.savefig('{}/accuracy.png'.format(save_path))

    return ece

