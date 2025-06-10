import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
medianprops = {
        'color': 'blue',  # Цвет линии
        'linewidth': 2,  # Толщина линии
        'linestyle': '-'  # Стиль линии (сплошная)
    }
meanprops = {
    'color': 'green',      # Цвет линии среднего
    'linewidth': 2,        # Толщина линии
    'linestyle': '--'       # Стиль линии (пунктир)
}
time_df = pd.read_csv('full_times.csv')
metrics_dfs = {}
'''
dic = {'prec@10': {'KNN': [], 'Random': [], 'Election_series_SNTV': [], 'Election_remove_bad_SNTV': [], 'Election_series_BnB': [], 'Election_remove_bad_BnB': []},
       'recall@10': {'KNN': [], 'Random': [], 'Election_series_SNTV': [], 'Election_remove_bad_SNTV': [], 'Election_series_BnB': [], 'Election_remove_bad_BnB': []},
       'ndcg': {'KNN': [], 'Random': [], 'Election_series_SNTV': [], 'Election_remove_bad_SNTV': [], 'Election_series_BnB': [], 'Election_remove_bad_BnB': []},
       'serendipity@10': {'KNN': [], 'Random': [], 'Election_series_SNTV': [], 'Election_remove_bad_SNTV': [], 'Election_series_BnB': [], 'Election_remove_bad_BnB': []}}
for num, key in enumerate(['prec@10', 'recall@10', 'ndcg', 'serendipity@10']):
    for i in range(41):
        metrics_dfs[i] = pd.read_csv('metrics3/metrics_random_cut_'+str(i)+'.csv')
        for column in metrics_dfs[i].columns:
            if column != 'Unnamed: 0' and column != 'Popular':
                #print(column)
                dic[key][column].append(metrics_dfs[i].at[num, column] - metrics_dfs[i].at[num, 'Popular'])
                #print(metrics_dfs[i].at[num, column])
    fig, ax = plt.subplots(figsize=(16, 9))
    bp = ax.boxplot(dic[key].values(), showmeans=True, meanline=True, medianprops=medianprops, meanprops=meanprops)
    ax.set_xticklabels(['KNN', 'Random', 'series (SNTV)', 'one election (SNTV)', 'series (BnB)', 'one election (BnB)'])
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.set_title(key + ' по сравнению с Popular')
    #ax.set_xlabel('Алгоритмы')
    #ax.set_ylabel(key)
    ax.legend([bp['medians'][0], bp['means'][0], ax.lines[-1]],
              ['Медиана', 'Среднее', 'Значение Popular'],
              loc='upper right')
    plt.savefig('boxplot_BnB_' + key + '.png', dpi=300, bbox_inches='tight')
    plt.show()

metrics_dfs = {}
dic = {'prec@10': {'KNN': [], 'Random': [], 'Election_series': [], 'Election_remove_bad': []},
       'recall@10': {'KNN': [], 'Random': [], 'Election_series': [], 'Election_remove_bad': []},
       'ndcg': {'KNN': [], 'Random': [], 'Election_series': [], 'Election_remove_bad': []},
       'serendipity@10': {'KNN': [], 'Random': [], 'Election_series': [], 'Election_remove_bad': []}}
for num, key in enumerate(['prec@10', 'recall@10', 'ndcg', 'serendipity@10']):
    for i in range(100):
        metrics_dfs[i] = pd.read_csv('metrics2/metrics_random_cut_'+str(i)+'.csv')
        for column in metrics_dfs[i].columns:
            if column != 'Unnamed: 0' and column != 'Popular':
                #print(column)
                dic[key][column].append(metrics_dfs[i].at[num, column] - metrics_dfs[i].at[num, 'Popular'])
                #print(metrics_dfs[i].at[num, column])
    fig, ax = plt.subplots(figsize=(16, 9))
    bp = ax.boxplot(dic[key].values(), showmeans=True, meanline=True, medianprops=medianprops, meanprops=meanprops)
    ax.set_xticklabels(['KNN', 'Random', 'series (SNTV)', 'one election (SNTV)'])
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.set_title(key + ' по сравнению с Popular')
    #ax.set_xlabel('Алгоритмы')
    #ax.set_ylabel(key)
    ax.legend([bp['medians'][0], bp['means'][0], ax.lines[-1]],
              ['Медиана', 'Среднее', 'Значение Popular'],
              loc='upper right')
    plt.savefig('boxplot_' + key + '.png', dpi=300, bbox_inches='tight')
    plt.show()
'''


fig, ax = plt.subplots(figsize=(16, 9))
indices = np.arange(5)
width = 0.2
colors = ['red', 'blue', 'blue', 'green', 'green']
linestyles = ['-', '--', '-.', ':', '-']
times_list = time_df.loc[[0, 1, 2, 4, 6], 'value'].tolist()
print(times_list)

ax.bar(indices, times_list, color = colors, width = width)
labs = ['KNN', 'Random', 'Popular', 'series (SNTV)', 'one election (SNTV)']
#ax.set_xlabel('Категории')
ax.set_ylabel('seconds')
ax.set_title('Среднее время создания рекомендаций')
ax.set_xticks(indices)
ax.set_xticklabels(labs)
#ax.legend()
plt.savefig('full_times_nolongs_bar.png', dpi=300, bbox_inches='tight')
plt.show()

'''
zero_cut_df = pd.read_csv('metrics_zero_cut.csv')
num_rows = 5

labs = ['KNN', 'Random', 'Popular', 'series (SNTV)', 'one election (SNTV)', 'series (BnB)', 'one election (BnB)']
indices = np.arange(7)
width = 0.2
colors = ['red', 'blue', 'blue', 'green', 'green', 'purple', 'purple']
#linestyles = ['-', '--', '-.', ':', '-']
for i, (index, row) in enumerate(zero_cut_df.iterrows()):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.grid(True)
    #print(index)
    ax.bar(indices, row.values[1:], color = colors, width = width)
    ax.set_title(row.values[0])
    # ax.set_xlabel('Категории')
    # ax.set_ylabel('Значения')
    ax.set_xticks(indices)
    ax.set_xticklabels(labs)
    plt.tight_layout()
    plt.savefig(row.values[0] + '_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
'''
# ax.set_xlabel('Категории')
# ax.set_ylabel('Значения')
# ax.set_title('Гистограммы по строкам DataFrame')
# ax.set_xticks(indices + width * (len(time_df.columns) - 1) / 2)
# ax.set_xticklabels(time_df.columns)
# ax.legend()

