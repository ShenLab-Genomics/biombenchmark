import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
X_LABEL_SIZE = 8
X_TICK_SIZE = 7


def draw_nRC(finput):
    # Load the data
    data = pd.read_csv(finput, sep=',', header=0)
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

    plt.xlabel('Method', size=X_LABEL_SIZE)
    plt.ylabel('Time cost (relative)', size=X_LABEL_SIZE)
    plt.bar(time_cost.index, time_cost.values /
            time_cost.values.max(), color='skyblue')
    plt.xticks(fontsize=X_TICK_SIZE)

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


def draw_m6A(finput1, finput2):
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

        plt.plot(method_data['Epoch'], method_data['f1s'], label=method)

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

    plt.savefig('m6ACls_performance.png', dpi=150)


if __name__ == '__main__':
    # draw_nRC('nRCcls_collected_data.csv')
    draw_m6A('m6A_short_collected_data.csv', 'm6A_long_collected_data.csv')
