import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

label_size = {'xsmall': 5, 'small': 6, 'medium': 7, 'large': 8}


def draw_epoch_performance(data: pd.DataFrame,
                           column='f1s',
                           skip_method=['nRC', 'RNACon'],
                           method_color={},
                           general_color={}):
    methods = data['Method'].unique()
    for method in methods:
        if method in skip_method:
            continue
        method_data = data[data['Method'] == method]
        color = method_color[method] if method in method_color else general_color['general']
        plt.plot(method_data['Epoch'],
                 method_data[column],
                 linewidth=1,
                 label=method,
                 color=color, alpha=0.8)
    plt.xticks(fontsize=label_size['medium'])
    plt.yticks(fontsize=label_size['medium'])
    plt.ylim(0, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def load_nRC_data(file_list, param_list, column='f1s'):
    assert len(file_list) == len(param_list)
    data = []
    for i in range(len(file_list)):
        df = pd.read_csv(file_list[i])
        df['param'] = param_list[i]
        data.append(df)
    # for each method, use the parameter with the best performance
    data = pd.concat(data)
    data = data.reset_index(drop=True)
    result = []
    param_dict = {}
    for method in data['Method'].unique():
        method_data = data[data['Method'] == method]
        if isinstance(column, list) and (len(column) > 1):
            best_param = method_data.loc[method_data[column].mean(
                axis=1).idxmax()]['param']
        else:
            best_param = method_data.loc[method_data[column].idxmax()]['param']
        result.append(method_data[method_data['param'] == best_param])
        param_dict[method] = best_param
    result = pd.concat(result)
    return result, param_dict


def draw_epoch_performance_m6A(data,
                               column='f1s',
                               legend=False,
                               method_color={},
                               general_color={}):
    X_TICK_SIZE = label_size['medium']
    X_LABEL_SIZE = label_size['large']
    methods = data['Method'].unique()
    for method in methods:
        if method in ['nRC', 'RNACon']:
            continue
        method_data = data[data['Method'] == method]
        # line_style = '-' if '(Long)' in method else '--'
        line_style = '-' if '(Long)' in method else '-'
        method_name = method.strip('(Long)')
        color = method_color[method_name] if method_name in method_color else general_color['']
        plt.plot(method_data['Epoch']+1,
                 method_data[column],
                 line_style,
                 linewidth=1,
                 color=color,
                 label=method_name,
                 alpha=0.8)

    plt.xticks(np.arange(1, 11, 1), fontsize=X_TICK_SIZE)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=X_TICK_SIZE)
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.yticks(fontsize=X_TICK_SIZE)
    plt.xticks(fontsize=X_TICK_SIZE)

    if legend:
        plt.legend(prop={'size': X_TICK_SIZE},
                   loc='lower right', ncol=1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def draw_m6a_supp(fig, data, legend=True):
    X_TICK_SIZE = label_size['medium']
    X_LABEL_SIZE = label_size['large']
    gs1 = fig.add_gridspec(nrows=2, ncols=3)
    #

    def template(data, column, label):
        draw_epoch_performance_m6A(data, column=column)
        plt.ylabel(label, size=X_LABEL_SIZE)
        plt.xlabel('Number of epochs', size=X_LABEL_SIZE)
        plt.ylim(0.3, 0.9)

    ax1 = fig.add_subplot(gs1[0, 0])
    template(data, 'accuracy', 'Accuracy')

    ax2 = fig.add_subplot(gs1[0, 1])
    template(data, 'recall', 'Recall')

    ax3 = fig.add_subplot(gs1[0, 2])
    template(data, 'precision', 'Precision')

    ax4 = fig.add_subplot(gs1[1, 0])
    template(data, 'pr_auc', 'PR-AUC')

    ax5 = fig.add_subplot(gs1[1, 1])
    template(data, 'auc', 'ROC-AUC')

    ax6 = fig.add_subplot(gs1[1, 2])
    template(data, 'f1s', 'F1-score')

    plt.legend(prop={'size': X_TICK_SIZE-2},
               loc=(1.1, 0.0))
