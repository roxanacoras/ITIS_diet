# model
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
# stats
from statsmodels.stats.contingency_tables import mcnemar
# utils
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# plotting
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def khatri_rao(df1, df2):
    
    """
    columns must match
    resturns katri_rho
    """
    
    krdf = []
    for factor in df1.columns:
        a = df1[[factor]].T
        b = df2[[factor]].T
        kr_tmp = pd.DataFrame(np.kron(a, b),
                              columns=pd.MultiIndex.from_product([a, b])).T
        kr_tmp.columns = [factor]
        krdf.append(kr_tmp)
    
    return pd.concat(krdf, axis=1)

def mcnemar_test(contingency_tables):

    mcnemar_stats = []
    for i in range(len(contingency_tables[0])):
        # where true both
        c1_tf = contingency_tables[0][i]
        c2_tf = contingency_tables[1][i]
        # both true
        tt = sum(c1_tf &  c2_tf)
        # C1 true & c2 False
        tf = sum(c1_tf &  ~c2_tf)
        # C1 false & c2 true
        ft = sum(~c1_tf &  c2_tf)
        # both false
        ff = sum(~c1_tf &  ~c2_tf)
        # contingency
        contingency_i = np.array([[tt, tf], [ft, ff]])
        # calculate mcnemar test
        result = mcnemar(contingency_i, exact=True)
        # summarize the finding
        mcnemar_stats.append([result.statistic, result.pvalue])

    mcnemar_stats = pd.DataFrame(mcnemar_stats,
                                 columns=['test-statistic', 'P'])
    mcnemar_stats.index.name = 'N-fold'

    return mcnemar_stats

def roc_binary(Xdfs, ydf,
               classifier,
               labels,
               color_map,
               number_of_splits,
               ax):

    # subset
    X_1 = Xdfs[0].values
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(ydf.values)
    n_samples = X_1.shape[0]
    # set state
    random_state = np.random.RandomState(42)
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=number_of_splits)
    # build list too fill
    tprs = [[] for i in range(len(Xdfs))]
    aucs = [[] for i in range(len(Xdfs))]
    contingencys = [[] for i in range(len(Xdfs))]
    mean_fpr = np.linspace(0, 1, n_samples)
    # run CV
    i = 0
    for train, test in cv.split(X_1, y):
        for (x_j, X), cmean  in zip(enumerate([x.values for x in Xdfs]), color_map):
            # for each X
            class_tmp = classifier.fit(X[train], y[train])
            probas_ = class_tmp.predict_proba(X[test])
            ppreds_ = class_tmp.predict(X[test])
            correct_ = ppreds_ == y[test]
            contingencys[x_j].append(correct_)
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs[x_j].append(interp(mean_fpr, fpr, tpr))
            tprs[x_j][-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs[x_j].append(roc_auc)
            #plt.plot(fpr, tpr, lw=1., c=cmean, alpha=0.3)

            i += 1
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray',
             label='Chance', alpha=.5)

    for x_j, cmean in enumerate(color_map):
        label = labels[x_j]
        mean_tpr = np.mean(tprs[x_j], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs[x_j])
        ax.plot(mean_fpr, mean_tpr, color=cmean,
                 label=r'%s %0.2f $\pm$ %0.2f' % (label, mean_auc, std_auc),
                 lw=2, alpha=.8)
        #label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc),
        std_tpr = np.std(tprs[x_j], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=cmean, alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', weight='bold',
                  fontsize=22, color='black')
    ax.set_ylabel('True Positive Rate', weight='bold',
                  fontsize=22, color='black')
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(0, 1)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('grey')
    # make a legend
    handles, labels = ax.get_legend_handles_labels()
    labels = list(labels)[1:len(Xdfs) + 1]
    handles = list(handles)[1:len(Xdfs) + 1]
    legend = ax.legend(handles, labels, loc=2, 
                            bbox_to_anchor=(0.95, .45),
                            prop={'size':16,
                                  'weight':'bold'},
                            handlelength=0, handletextpad=0,
                            fancybox=True, framealpha=0.0, 
                            ncol=1, markerscale=3,
                            facecolor="white",
                            title="AUC $\pm$ std.")
    legend.get_title().set_fontsize('12')
    legend.get_title().set_ha('right')
    legend.get_title().set_position((-135, 0.2))
    for item in legend.legendHandles:
        item.set_visible(False)
    for text, c in zip(legend.get_texts(), color_map):
        text.set_color(c)
        text.set_alpha(1.0)
        text.set_ha('right')
    for sp_i in ax.spines.values():
        sp_i.set_linewidth(3)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('black')
    ax.tick_params(axis='y', colors='black', width=4, length=10)
    ax.tick_params(axis='x', colors='black', width=4, length=10)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties('arial')
        tick.set_weight("bold")
        tick.set_color("black")
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties('arial')
        tick.set_weight("bold")
        tick.set_color("black")
        tick.set_fontsize(14)

    return ax, contingencys

def roc_multiclass(Xdfs, ydf,
                   classifier,
                   labels,
                   color_map,
                   number_of_splits,
                   ax):

    # subset
    X_1 = Xdfs[0].values
    n_classes = len(set(ydf))
    y = label_binarize(ydf.values, classes=list(set(ydf)))
    n_samples = X_1.shape[0]
    # set state
    random_state = np.random.RandomState(42)
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=number_of_splits)
    # build list too fill

    
    fprs = [{'micro':[], 'macro':[], 0:[], 1:[], 2:[]} for i in range(len(Xdfs))]
    tprs = [{'micro':[], 'macro':[], 0:[], 1:[], 2:[]} for i in range(len(Xdfs))]
    aucs = [{'micro':[], 'macro':[], 0:[], 1:[], 2:[]} for i in range(len(Xdfs))]
    contingencys = [[] for i in range(len(Xdfs))]
    mean_fpr = np.linspace(0, 1, n_samples)
    # run CV
    for i_fold, (train, test) in enumerate(cv.split(X_1, y.argmax(1))):
    #for train, test in cv.split(X_1, y):
        for (x_j, X), cmean  in zip(enumerate([x.values for x in Xdfs]), color_map):
            # for each X
            class_tmp = classifier.fit(X[train], y[train])
            probas_ = class_tmp.predict_proba(X[test])
            ppreds_ = class_tmp.predict(X[test])
            probas_ = np.array([p_[:, 1] for p_ in probas_]).T
            # AUCs per class
            for class_i in range(n_classes):
                fpr_tmp, tprs_tmp, _ = roc_curve(y[test][:, class_i], probas_[:, class_i])
                fprs[x_j][class_i].append(fpr_tmp)
                tprs[x_j][class_i].append(tprs_tmp)
                aucs[x_j][class_i].append(auc(fpr_tmp, tprs_tmp))

            # Compute micro-average ROC curve and ROC area
            fpr_tmp, tpr_tmp, _ = roc_curve(y[test].ravel(), probas_.ravel())
            fprs[x_j]["micro"].append(fpr_tmp)
            tprs[x_j]["micro"].append(tpr_tmp)
            tprs[x_j]["micro"][-1][0] = 0.0
            aucs[x_j]["micro"].append(auc(fprs[x_j]["micro"][i_fold],
                                          tprs[x_j]["micro"][i_fold]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(mean_fpr)
            for class_i in range(n_classes):
                mean_tpr += interp(mean_fpr,
                                   fprs[x_j][class_i][i_fold],
                                   tprs[x_j][class_i][i_fold])
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            # add classes to dict
            fprs[x_j]["macro"].append(mean_fpr)
            tprs[x_j]["macro"].append(mean_tpr)
            tprs[x_j]["macro"][-1][0] = 0.0
            aucs[x_j]["macro"].append(auc(fprs[x_j]["macro"][i_fold],
                                          tprs[x_j]["macro"][i_fold]))
            # get macro cont.
            correct_ = (ppreds_ == y[test]).all(1)
            contingencys[x_j].append(correct_)
            # Compute ROC curve and area the curve
            #plt.plot(fprs[x_j]["macro"][i_fold],
            #         tprs[x_j]["macro"][i_fold],
            #         lw=1., c=cmean, alpha=0.3)

            #i += 1
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray',
             label='Chance', alpha=.5)

    for x_j, cmean in enumerate(color_map):
        label = labels[x_j]
        all_fold_tpr = np.array(tprs[x_j]["macro"])
        mean_tpr = np.mean(all_fold_tpr, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs[x_j]["micro"])
        ax.plot(mean_fpr, mean_tpr, color=cmean,
                 label=r'%s %0.2f $\pm$ %0.2f' % (label, mean_auc, std_auc),
                 lw=2, alpha=.8)
        #label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc),
        std_tpr = np.std(tprs[x_j]["macro"], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=cmean, alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', weight='bold',
                  fontsize=22, color='black')
    ax.set_ylabel('True Positive Rate', weight='bold',
                  fontsize=22, color='black')
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(0, 1)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('grey')
    # make a legend
    handles, labels = ax.get_legend_handles_labels()
    labels = list(labels)[1:len(Xdfs) + 1]
    handles = list(handles)[1:len(Xdfs) + 1]
    legend = ax.legend(handles, labels, loc=2, 
                            bbox_to_anchor=(0.95, .35),
                            prop={'size':16,
                                  'weight':'bold'},
                            handlelength=0, handletextpad=0,
                            fancybox=True, framealpha=0.0, 
                            ncol=1, markerscale=3,
                            facecolor="white",
                            title="AUC $\pm$ std.")
    legend.get_title().set_fontsize('12')
    legend.get_title().set_ha('right')
    legend.get_title().set_position((-135, 0.2))
    for item in legend.legendHandles:
        item.set_visible(False)
    for text, c in zip(legend.get_texts(), color_map):
        text.set_color(c)
        text.set_alpha(1.0)
        text.set_ha('right')
    for sp_i in ax.spines.values():
        sp_i.set_linewidth(3)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('black')
    ax.tick_params(axis='y', colors='black', width=4, length=10)
    ax.tick_params(axis='x', colors='black', width=4, length=10)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties('arial')
        tick.set_weight("bold")
        tick.set_color("black")
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties('arial')
        tick.set_weight("bold")
        tick.set_color("black")
        tick.set_fontsize(14)

    return ax, contingencys

