import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'
plt.rcParams['font.sans-serif'] = 'Arial'
global_xtick_size = {'small': 6, 'medium': 7, 'large': 8}
X_LABEL_SIZE = 8
X_TICK_SIZE = 7


def draw_nRC(finput):
    # Load the data
    data = pd.read_csv(finput, sep=',', header=0)
    print(data.head())

    # Plot the data

    fig = plt.figure(figsize=(6.7, 5), dpi=150, constrained_layout=True)
    gs1 = fig.add_gridspec(nrows=2, ncols=6)

    ax1 = fig.add_subplot(gs1[0, 0:3])
    plt.xlabel('Epoch', size=X_LABEL_SIZE)
    plt.ylabel('F1-score', size=X_LABEL_SIZE)
    methods = data['method'].unique()
    for method in methods:
        if method in ['nRC', 'RNACon']:
            continue
        method_data = data[data['method'] == method]

        plt.plot(method_data['Epoch'], method_data['f1s'], label=method)

    plt.legend(prop={'size': X_TICK_SIZE},
               loc='lower right')

    # Plot the average time cost
    ax2 = fig.add_subplot(gs1[0, 3:])

    # Convert the time to float
    data['Time'] = data['Time'].apply(lambda x: float(str(x).split('s')[0]))
    # Calculate the average time cost
    time_cost = data.groupby('method')['Time'].mean()
    time_cost = time_cost[~time_cost.index.isin(['nRC', 'RNACon'])]
    time_cost.sort_values(ascending=False, inplace=True)
    plt.xlabel('Method', size=X_LABEL_SIZE)
    plt.ylabel('Time cost (relative)', size=X_LABEL_SIZE)
    plt.bar(time_cost.index, time_cost.values /
            time_cost.values.max(), color='skyblue')
    plt.xticks(fontsize=X_TICK_SIZE, ha='right', rotation=45)

    # Select the best epoch for each method (based on F1-score)
    # Then, plot the precision, recall, and F1-score of the best epoch

    performance = []
    for method in methods:
        method_data = data[data['method'] == method]
        best_epoch = method_data.loc[method_data['f1s'].idxmax()]
        performance.append(best_epoch)
    performance = pd.DataFrame(performance)
    performance.set_index('method', inplace=True)
    performance.sort_values('f1s', ascending=True, inplace=True)

    ax3 = fig.add_subplot(gs1[1, 0:2])
    plt.barh(performance.index, performance['precision'],
             height=0.65, label=method)
    # display the value on the bar
    for i, v in enumerate(performance['precision']):
        plt.text(v - 0.02, i, '{:.4f}'.format(v), color='white',
                 va='center', ha='right', fontsize=X_TICK_SIZE)
    plt.xlabel('Precision', size=X_LABEL_SIZE)
    plt.ylabel('Method', size=X_LABEL_SIZE)
    plt.xticks(fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=X_TICK_SIZE)

    ax4 = fig.add_subplot(gs1[1, 2:4])
    plt.barh(performance.index,
             performance['recall'], height=0.65, label=method)
    # display the value on the bar
    for i, v in enumerate(performance['recall']):
        plt.text(v - 0.02, i, '{:.4f}'.format(v), color='white',
                 va='center', ha='right', fontsize=X_TICK_SIZE)
    plt.xlabel('Recall', size=X_LABEL_SIZE)
    plt.ylabel(None, size=X_LABEL_SIZE)
    plt.xticks(fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=X_TICK_SIZE)

    ax5 = fig.add_subplot(gs1[1, 4:])
    plt.barh(performance.index, performance['f1s'], height=0.65, label=method)
    # display the value on the bar
    for i, v in enumerate(performance['f1s']):
        plt.text(v - 0.02, i, '{:.4f}'.format(v), color='white',
                 va='center', ha='right', fontsize=X_TICK_SIZE)
    plt.xlabel('F1-score', size=X_LABEL_SIZE)
    plt.ylabel('', size=X_LABEL_SIZE)
    plt.xticks(fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=X_TICK_SIZE)

    plt.savefig('nRCcls_performance.png', dpi=150)
    plt.savefig('nRCcls_performance.pdf', dpi=150)


def draw_m6A(finput1, finput2, output):
    # Load the data
    data_short = pd.read_csv(finput1, sep=',', header=0)
    data_long = pd.read_csv(finput2, sep=',', header=0)
    # data_short['method'] = data_short['method'] + '(short)'
    data_long['method'] = data_long['method'] + '(long)'
    data = pd.concat([data_short, data_long], axis=0)
    print(data.head())

    # Plot the data

    fig = plt.figure(figsize=(8, 5), dpi=150, constrained_layout=True)
    gs1 = fig.add_gridspec(nrows=2, ncols=6)

    ax1 = fig.add_subplot(gs1[0, 0:3])
    plt.xlabel('Epoch', size=X_LABEL_SIZE)
    plt.ylabel('F1-score', size=X_LABEL_SIZE)
    methods = data['method'].unique()
    for method in methods:
        if method in ['nRC', 'RNACon']:
            continue
        method_data = data[data['method'] == method]

        plt.plot(method_data['Epoch']+1, method_data['f1s'], label=method)
    plt.xticks(np.arange(1, 11, 1), fontsize=X_TICK_SIZE)
    plt.xlim(1, 10)

    plt.legend(prop={'size': X_TICK_SIZE},
               loc='lower right')

    # Plot the average time cost
    ax2 = fig.add_subplot(gs1[0, 3:])
    # Convert the time to float
    data['Time'] = data['Time'].apply(lambda x: float(str(x).split('s')[0]))
    # Calculate the average time cost
    time_cost = data.groupby('method')['Time'].mean()
    time_cost.sort_values(ascending=False, inplace=True)
    time_cost = time_cost[~time_cost.index.isin(['nRC', 'RNACon'])]

    plt.xlabel('Method', size=X_LABEL_SIZE)
    plt.ylabel('Time cost (relative)', size=X_LABEL_SIZE)
    plt.bar(time_cost.index, time_cost.values /
            time_cost.values.max(), color='skyblue')
    plt.xticks(fontsize=X_TICK_SIZE, rotation=45, ha='right')

    performance = []
    for method in methods:
        method_data = data[data['method'] == method]
        best_epoch = method_data.loc[method_data['f1s'].idxmax()]
        performance.append(best_epoch)
    performance = pd.DataFrame(performance)
    performance.set_index('method', inplace=True)
    performance.sort_values('f1s', ascending=True, inplace=True)

    ax3 = fig.add_subplot(gs1[1, 0:3])
    plt.barh(performance.index, performance['f1s'],
             height=0.65, label=method, color='#8E9BAE')
    # display the value on the bar
    for i, v in enumerate(performance['f1s']):
        plt.text(v - 0.02, i, '{:.4f}'.format(v), color='white',
                 va='center', ha='right', fontsize=X_TICK_SIZE)
    plt.xlabel('F1-score', size=X_LABEL_SIZE)
    plt.ylabel('Method', size=X_LABEL_SIZE)
    plt.xticks(fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=X_TICK_SIZE)

    ax4 = fig.add_subplot(gs1[1, 3:])
    plt.barh(performance.index, performance['pr_auc'],
             height=0.65, label=method, color='#8E9BAE')
    # display the value on the bar
    for i, v in enumerate(performance['pr_auc']):
        plt.text(v - 0.02, i, '{:.4f}'.format(v), color='white',
                 va='center', ha='right', fontsize=X_TICK_SIZE)
    plt.xlabel('PR-AUC', size=X_LABEL_SIZE)
    plt.ylabel(None, size=X_LABEL_SIZE)
    plt.xticks(fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=X_TICK_SIZE)

    plt.savefig(output, dpi=150)


def draw_model_stat(finput):
    trainable_params = {
        'RNAFM': 99522828,
        'RNAMSM': 95921177,
        'RNAErnie': 87230978,
        'RNABERT': 478682,
        'DNABERT': 86095874,
        'DNABERT2': 117070082,
        'SpliceBERT': 19447298,
    }

    data = pd.read_csv(finput, sep=',', header=0)
    print(data.head())
    # Convert the time to float
    data['Time'] = data['Time'].apply(lambda x: float(str(x).split('s')[0]))

    print(data)

    fig = plt.figure(figsize=(5, 3), dpi=150, constrained_layout=True)
    for method in data['method'].unique():
        if method not in trainable_params:
            continue
        method_data = data[data['method'] == method]
        method_time = method_data['Time'].mean()
        method_f1s = method_data['f1s'].mean()
        method_params = trainable_params[method]

        plt.scatter(method_time, method_f1s, s=method_params/1e5,
                    label=method, alpha=0.6)

    plt.legend(loc='lower right', prop={'size': X_TICK_SIZE})
    plt.xlim(0, 35)
    plt.ylim(0, 1)
    plt.xlabel('Time cost (s/epoch)')
    plt.ylabel('F1-score')
    plt.title('Scale of trainable parameters', size=X_LABEL_SIZE)
    plt.savefig('model_stat.png', dpi=150)
    pass


METHOD_COLOR = {
    'RNAFM': '#8D91AA',
    'SpliceBERT': '#78677A',
    'RNAMSM': '#E7ADAC',
    'RNAErnie': '#013E41',
    'SpTransformer': '#5B7493',
    '': '#9AA2AD',
        '': '#EBC1A8',
}


def draw_splice_stat(finput):

    data = pd.read_csv(finput, sep=',', header=0)
    print(data.head())
    # Convert the time to float
    data['Time'] = data['Time'].apply(lambda x: float(str(x).split('s')[0]))

    fig = plt.figure(figsize=(5, 4), dpi=150, constrained_layout=True)
    gs1 = fig.add_gridspec(nrows=2, ncols=2)

    ax1 = fig.add_subplot(gs1[0, 0])
    methods = data['Method'].unique()

    data['Epoch'] = data['Epoch'].str.split(':').apply(lambda x: int(x[1]))
    for idx, method in enumerate(methods):
        method_data = data[data['Method'] == method]

        columns = [f'pr_auc_{i}' for i in range(15)]

        avg_pr_auc = method_data[columns].mean(axis=1)
        plt.plot(method_data['Epoch'] + 1, avg_pr_auc, label=method,
                 color=METHOD_COLOR[method])
    plt.xticks(np.arange(1, 6, 1), fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=X_TICK_SIZE)
    plt.xlabel('Epoch', size=X_LABEL_SIZE)
    plt.ylabel('Average PR-AUC', size=X_LABEL_SIZE)

    # show time cost
    ax2 = fig.add_subplot(gs1[0, 1])
    # for idx, method in enumerate(methods):
    # method_data = data[data['Method'] == method]
    time_cost = data.groupby('Method')['Time'].mean()
    time_cost.sort_values(ascending=False, inplace=True)
    print(time_cost)
    plt.bar(time_cost.index, time_cost.values /
            time_cost.values.max(), width=0.4, color='skyblue')
    plt.xticks(fontsize=X_TICK_SIZE, rotation=45, ha='right')
    plt.yticks(fontsize=X_TICK_SIZE)
    plt.ylabel('Time cost (relative)', size=X_LABEL_SIZE)

    ax3 = fig.add_subplot(gs1[1, :])

    width = 0.6 / len(methods)
    for idx, method in enumerate(methods):
        method_data = data[data['Method'] == method]

        pr_auc = []
        for i in range(15):
            pr_auc.append(
                method_data[f'pr_auc_{i}'].max())
        print(pr_auc)
        plt.bar(np.arange(15)+(idx - len(methods)//2) *
                width, pr_auc, width=width, label=method, color=METHOD_COLOR[method])

        # plt.plot(method_data['Epoch'], method_data['f1s'], label=method)
    plt.legend(loc='lower right', prop={'size': X_TICK_SIZE})
    tissue_classes = ['Adipose Tissue', 'Blood', 'Blood Vessel', 'Brain', 'Colon', 'Heart', 'Kidney',
                      'Liver', 'Lung', 'Muscle', 'Nerve', 'Small Intestine', 'Skin', 'Spleen', 'Stomach']
    plt.xticks(np.arange(15), tissue_classes,
               fontsize=X_TICK_SIZE, rotation=45, ha='right')
    plt.yticks(fontsize=X_TICK_SIZE)
    plt.xlabel('Tissue class', size=X_LABEL_SIZE)
    plt.ylabel('PR-AUC', size=X_LABEL_SIZE)

    plt.savefig('splice15cls_performance.png', dpi=150)
    plt.savefig('splice15cls_performance.pdf', dpi=150)
    pass


def draw_splice3_stat(finput):
    data = pd.read_csv(finput, sep=',', header=0)
    print(data.head())
    # Convert the time to float
    data['Time'] = data['Time'].apply(lambda x: float(str(x).split('s')[0]))

    fig = plt.figure(figsize=(5, 4), dpi=150, constrained_layout=True)
    gs1 = fig.add_gridspec(nrows=2, ncols=2)

    ax1 = fig.add_subplot(gs1[0, 0])
    methods = data['Method'].unique()

    data['Epoch'] = data['Epoch'].astype(int)
    for idx, method in enumerate(methods):
        method_data = data[data['Method'] == method]

        columns = [f'pr_auc_{i}' for i in range(2)]

        avg_pr_auc = method_data[columns].mean(axis=1)
        plt.plot(method_data['Epoch'] + 1, avg_pr_auc, label=method,
                 color=METHOD_COLOR[method])
    plt.xticks(np.arange(1, 6, 1), fontsize=X_TICK_SIZE)
    plt.yticks(fontsize=X_TICK_SIZE)
    plt.xlabel('Epoch', size=X_LABEL_SIZE)
    plt.ylabel('Average PR-AUC', size=X_LABEL_SIZE)

    # show time cost
    ax2 = fig.add_subplot(gs1[0, 1])
    # for idx, method in enumerate(methods):
    # method_data = data[data['Method'] == method]
    time_cost = data.groupby('Method')['Time'].mean()
    time_cost.sort_values(ascending=False, inplace=True)
    print(time_cost)
    plt.bar(time_cost.index, time_cost.values /
            time_cost.values.max(), width=0.4, color='skyblue')
    plt.xticks(fontsize=X_TICK_SIZE, rotation=45, ha='right')
    plt.yticks(fontsize=X_TICK_SIZE)
    plt.ylabel('Time cost (relative)', size=X_LABEL_SIZE)

    plt.savefig('splice3cls_performance.png', dpi=150)
    plt.savefig('splice3cls_performance.pdf', dpi=150)

if __name__ == '__main__':
    # draw_nRC('nRCcls_collected_data.csv')
    # draw_m6A('m6A_short_miclip_collected_data.csv',
    #          'm6A_long_miclip_collected_data.csv',
    #          'm6ACls_performance.png')

    # draw_m6A('m6A_short_seq_collected_data.csv',
    #          'm6A_long_seq_collected_data.csv',
    #          'm6ACls_performance_seq.png')

    # draw_model_stat('nRCcls_collected_data.csv')

    # draw_splice_stat('splice15cls_collected_data.csv')

    draw_splice3_stat('splice3cls_collected_data.csv')
