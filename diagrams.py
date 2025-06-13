import numpy as np
import pandas as pd
from itertools import product
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.pyplot import figure
from rich.columns import Columns
from rectools import Columns

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

def bar_metrics_draw(labs, list, name, title):
    fig, ax = plt.subplots(figsize=(16, 9))
    n = len(list)
    indices = np.arange(n)
    width = 0.2
    colors = ['green'] * n
    print(list)
    print(labs)
    ax.bar(indices, list, color=colors, width=width)
    # ax.set_xlabel('Категории')
    ax.set_title(title)
    ax.set_xticks(indices)
    ax.set_xticklabels(labs)
    # ax.legend()
    plt.savefig(name + '.png', dpi=300, bbox_inches='tight')
    #plt.show()
def box_plot_metrics_draw(dic, xlabs, title, name, ref = None):
    fig, ax = plt.subplots(figsize=(16, 9))
    bp = ax.boxplot(dic.values(), showmeans=True, meanline=True, medianprops=medianprops, meanprops=meanprops)
    #print(xlabs)
    ax.set_xticklabels(xlabs)
    if ref is not None:
        ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        ax.set_title(title)
        # ax.set_xlabel('Алгоритмы')
        # ax.set_ylabel(key)
        ax.legend([bp['medians'][0], bp['means'][0], ax.lines[-1]],
                  ['Медиана', 'Среднее', 'Значение ' + ref],
                  loc='upper right')
    else:
        ax.set_title(title)
        # ax.set_xlabel('Алгоритмы')
        # ax.set_ylabel(key)
        ax.legend([bp['medians'][0], bp['means'][0]],
                  ['Медиана', 'Среднее'],
                  loc='upper right')
    plt.savefig(name + '.png', dpi=300, bbox_inches='tight')
    #plt.show()
def lab_title_make(param_id, p = 0):
    if param_id == 0:
        return p, 'алгоритма выборов'
    if param_id == 1:
        return p, 'метода генерации расстояний'
    elif param_id == 2:
        return 'n = ' + str(p), 'количества степеней одобрения'
    elif param_id == 3:
        return 'k = ' + str(p), 'количества рекомендаций'
    elif param_id == 4:
        return 'взвешанное среднее' * p + 'антирекомендации' * (1 - p), 'метода устранения демократии'
    elif param_id == 5:
        if p == 0:
            return 'одиночные выборы', 'количества выборов'
        else:
            return 'последовательные выборы (' + str(p) + ')', 'количества выборов'
    else:
        return None
def string_make(combination, param_id, p = 0, qey = True):
    if param_id == 0:
        if qey:
            return (combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(combination[3]) +
                    '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(combination[5]))
        else:
            return (str(p) + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(combination[3]) +
                    '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(combination[5]))
    elif param_id == 1:
        if qey:
            return (combination[0] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
        else:
            return (combination[0] + '_' + str(p) + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
    elif param_id == 2:
        if qey:
            return (combination[0] + '_' + combination[1] + '_size=' + str(combination[3]) +
                    '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(combination[5]))
        else:
            return (combination[0] + '_' + combination[1] + '_deg=' + str(p) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
    elif param_id == 3:
        if qey:
            return (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
        else:
            return (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(p) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                    combination[5]))
    elif param_id == 4:
        if qey:
            return (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) + '_rate=' + str(combination[5]))
        else:
            return (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * p + '_antirec_' * (1 - p) + 'rate=' + str(
                    combination[5]))
    elif param_id == 5:
        if qey:
            return (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]))
        else:
            return (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
            combination[3]) +
                      '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(p))

    return None
def metrics_draw_small(param_id, inner_param_grid):
    small_param_grid = deepcopy(inner_param_grid)
    #print(small_param_grid)
    df_dic = {}
    dic_params = {
        0:'rule',
        1:'dist_method',
        2:'degrees',
        3:'size',
        4:'weighted',
        5:'series_rate'}
    ps = small_param_grid[dic_params[param_id]]
    print('параметры', ps)
    small_param_grid[dic_params[param_id]] = [0]
    param_values = small_param_grid.values()
    #print(dic_params[param_id])
    dic = {'prec': {},
           'recall': {},
           'ndcg': {},
           'serendipity': {},
           'novelty': {}}
    for user in rating[Columns.User].unique()[:100]:

        filename = f"my_films/test1/metrics_user{user}.csv"
        if os.path.exists(filename):
            df_dic[user] = pd.read_csv(filename)

    labs = []
    for p in ps:
        labs.append(lab_title_make(param_id, p)[0])
    #print(labs)
    title_part = lab_title_make(param_id)[1]
    for num, key in enumerate(['prec', 'recall', 'ndcg', 'serendipity', 'novelty']):
        dic[key] = {}
        for p in ps:
            dic[key][p] = []
            #print(p)
            for combination in product(*param_values):
                #print(combination)
                #col_key = string_make(combination, param_id)
                #print(col_key)

                col = string_make(combination, param_id, p, False)
                print(col)
                for user, df in df_dic.items():
                    if col in df.columns:
                        v = df.at[num, col]
                        #print(v)
                        dic[key][p].append(v)

        name = 'my_films/' + dic_params[param_id] + '_plots/' + key + '-' + dic_params[param_id]
        title = 'Значение ' + key + ' в зависимости от ' + title_part
        box_plot_metrics_draw(dic[key], labs, title, name)
        for i in range(1, len(ps)):
            dic[key][ps[i]] = np.array(dic[key][ps[i]]) - np.array(dic[key][ps[0]])
        dic[key].pop(ps[0], None)
        name = 'my_films/' + dic_params[param_id] + '_plots/diff/' + key + '-no ' + dic_params[param_id] + '=' + str(ps[0])
        title = key + ': Сравнение' + ' с ' + str(labs[0])
        box_plot_metrics_draw(dic[key], labs[1:], title, name, str(labs[0]))

def metrics_draw(param_id, inner_param_grid):
    param_grid = deepcopy(inner_param_grid)
    #print(param_grid)
    df_dic = {}
    dic_params = {
        0:'rule',
        1:'dist_method',
        2:'degrees',
        3:'size',
        4:'weighted',
        5:'series_rate'}
    ps = param_grid[dic_params[param_id]]
    param_grid[dic_params[param_id]] = [0]
    param_values = param_grid.values()
    print(dic_params[param_id])
    dic = {'prec': {},
           'recall': {},
           'ndcg': {},
           'serendipity': {},
           'novelty': {}}
    for user in rating[Columns.User].unique()[:100]:

        filename = f"my_films/test1/metrics_user{user}.csv"
        if os.path.exists(filename):
            df_dic[user] = pd.read_csv(filename)

    labs = []
    for p in ps:
        labs.append(lab_title_make(param_id, p)[0])

    title_part = lab_title_make(param_id)[1]
    for num, key in enumerate(['prec', 'recall', 'ndcg', 'serendipity', 'novelty']):
        for combination in product(*param_values):
            skip_flag = False
            col_key = string_make(combination, param_id)
            print(col_key)
            dic[key][col_key] = {}
            for p in ps:
                col = string_make(combination, param_id, p, False)
                #print(col)
                user_list = []
                for user, df in df_dic.items():
                    if col in df.columns:
                        # print(df[col][num])
                        v = df.at[num, col]
                        #print(v)
                        user_list.append(v)
                        # print(df.at[num, col])
                        # print(user_list)
                nonzeros = np.count_nonzero(np.array(user_list))
                #print(nonzeros)
                if nonzeros == 0:
                    skip_flag = True
                    break
                else:
                    dic[key][col_key][p] = user_list
            if skip_flag:
                continue
            name = 'my_films/' + dic_params[param_id] + '_plots/' + key + '-' + dic_params[param_id] + '@' + col_key
            title = 'Значение ' + key + ' в зависимости от ' + title_part  + ' (' + col_key + ')'
            box_plot_metrics_draw(dic[key][col_key], labs, title, name)
            for i in range(1, len(ps)):
                dic[key][col_key][ps[i]] = np.array(dic[key][col_key][ps[i]]) - np.array(dic[key][col_key][ps[0]])
            dic[key][col_key].pop(ps[0], None)
            name = 'my_films/' + dic_params[param_id] + '_plots/diff/' + key + '-no ' + dic_params[param_id] + '=' + str(ps[0]) + '@' + col_key
            title = key + ': Сравнение' + ' с ' + str(labs[0]) + ' (' + col_key + ')'
            box_plot_metrics_draw(dic[key][col_key], labs[1:], title, name, str(labs[0]))
# rating = pd.read_csv('archive/ratings_small.csv')
# #print(rating)
# print('до', rating['movieId'].nunique(), rating['userId'].nunique())
# item_user_counts = rating.groupby('movieId')['userId'].nunique()
# valid_items = item_user_counts[item_user_counts > 2].index
# rating = rating[rating['movieId'].isin(valid_items)]
#
# user_item_counts = rating.groupby('userId')['movieId'].nunique()
# valid_users = user_item_counts[user_item_counts > 2].index
# rating = rating[rating['userId'].isin(valid_users)]
# print('после', rating['movieId'].nunique(), rating['userId'].nunique())
# movies = pd.read_csv('archive/links_small.csv')
# metadata = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
# #print(metadata.head(10))
# #print(movies.head(10))
#
# movies['original_title'] = metadata['original_title'].reindex(movies.index, fill_value='unknown')
# links_dic = dict(zip(movies['movieId'], movies['original_title']))

rating = pd.read_csv('long_my_films.csv')
movies = pd.read_csv('map_my_films.csv')
links_dic = movies[movies.columns[1]].to_dict()

all_params_grid = {'rule':['SNTV', 'STV_star', 'STV_basic', 'BnB'],
               'dist_method':['jaccar', 'cosine', 'cosine_hat', 'pearson', 'pearson_hat', 'spearman', 'spearman_hat', 'kendall', 'kendall_hat'],
               'degrees':[2, 3, 4, 5, 6, 7, 8],
               'size':[10, 15, 20, 25, 30],
               'weighted':[True, False],
               'series_rate':[0, 1, 2, 3]}
params_grid = {'rule':['SNTV', 'STV_basic', 'STV_star'],
               'dist_method':['jaccar', 'cosine', 'cosine_hat', 'pearson', 'pearson_hat', 'spearman', 'spearman_hat', 'kendall_hat', 'kendall'],
               'degrees':[2, 3, 4, 5, 6, 7, 8, 9, 10],
               'size':[5, 10, 15, 20, 25, 30],
               'weighted':[False, True],
               'series_rate':[0, 1, 2, 3]}
#metrics_draw(1, params_grid)

for i in range(0, 6):
    metrics_draw_small(i, params_grid)
    metrics_draw(i, params_grid)

exit()



for num, key in enumerate(['prec', 'recall', 'ndcg', 'serendipity@10', 'novelty']):
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

