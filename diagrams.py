import numpy as np
import pandas as pd
from itertools import product
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.pyplot import figure
from rich.columns import Columns
from rectools import Columns
from scipy import stats
import re

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
papka = 'my_films/test3/'
papka_ml = 'my_films/test_ML/'

def chi_square_test(sample1, sample2):
    """
    Критерий хи-квадрат Пирсона для сравнения распределений
    """
    # Объединяем выборки для получения всех уникальных значений
    all_values = sorted(set(sample1) | set(sample2))

    # Подсчитываем частоты для каждой выборки
    freq1 = [list(sample1).count(val) for val in all_values]
    freq2 = [list(sample2).count(val) for val in all_values]

    # Создаем таблицу сопряженности
    observed = np.array([freq1, freq2])

    # Проверяем условие применимости (ожидаемые частоты >= 5)
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    min_expected = np.min(expected)

    return {
        'statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'min_expected_freq': min_expected,
        'test_name': 'Chi-square test'
    }
def permutation_test(sample1, sample2, n_permutations=10000, stat_func=None):
    """
    Перестановочный тест для сравнения двух выборок
    """
    if stat_func is None:
        # По умолчанию используем разность средних
        stat_func = lambda x, y: np.mean(x) - np.mean(y)

    # Наблюдаемая статистика
    observed_stat = stat_func(sample1, sample2)

    # Объединяем выборки
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)

    # Генерируем перестановки
    permuted_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_sample1 = combined[:n1]
        perm_sample2 = combined[n1:]
        permuted_stats.append(stat_func(perm_sample1, perm_sample2))

    # Вычисляем p-value
    permuted_stats = np.array(permuted_stats)
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))

    return {
        'observed_statistic': observed_stat,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'test_name': 'Permutation test'
    }
def mini_comparison(sample1, sample2, metric = 'prec', alpha=0.05, alt = None):
    """Комплексное сравнение двух выборок"""
    comp_dic = {'greater':'>', 'less':'<', 'two-sided':'><'}
    # # Тест на нормальность
    # _, p_norm1 = stats.shapiro(sample1)
    # _, p_norm2 = stats.shapiro(sample2)
    if np.all((np.array(sample1) - np.array(sample2)) == 0):
        return 1, '(выборки равны)'
    if len(sample1) < 20:
        normal_dist = False
    else:
        _, p_norm1 = stats.normaltest(sample1)
        _, p_norm2 = stats.normaltest(sample2)
        #print(p_norm1, p_norm2)

        normal_dist = (p_norm1 > alpha) and (p_norm2 > alpha)

    # # Тест Левена на равенство дисперсий
    # _, p_var = stats.levene(sample1, sample2)
    #
    # equal_var = p_var > alpha


    if normal_dist:
        if alt is None:
            _, t_pvalue_greater = stats.ttest_rel(sample1, sample2, alternative='greater')
            _, t_pvalue_less = stats.ttest_rel(sample1, sample2, alternative='less')
            #t_p_value = min(t_pvalue_greater, t_pvalue_less)
            if t_pvalue_greater < t_pvalue_less:
                return t_pvalue_greater, '(парный критерий Стьюдента (>))'
            else:
                return t_pvalue_less, '(парный критерий Стьюдента (<))'
        else:
            _, t_pvalue = stats.ttest_rel(sample1, sample2, alternative=alt)
            return t_pvalue, '(парный критерий Стьюдента (' + comp_dic[alt] + '))'


    # Непараметрические тесты (всегда)
    if alt is None:
        wilcox_greater = stats.wilcoxon(sample1, sample2, alternative='greater')
        wilcox_less = stats.wilcoxon(sample1, sample2, alternative='less')
        #print(wilcox_less, wilcox_greater)
        if wilcox_greater[1] < wilcox_less[1]:
            return wilcox_greater[1], '(критерий Уилкоксона (>))'
        else:
            return wilcox_less[1], '(критерий Уилкоксона (<))'
    else:
        wilcox = stats.wilcoxon(sample1, sample2, alternative=alt)
        return wilcox[1], '(критерий Уилкоксона (' + comp_dic[alt] + '))'


def comprehensive_comparison(sample1, sample2, alpha=0.05, alt = None):
    """Комплексное сравнение двух выборок"""

    print("=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА ===")
    print(f"Выборка 1: среднее={np.mean(sample1):.3f}, std={np.std(sample1):.3f}, n={len(sample1)}")
    print(f"Выборка 2: среднее={np.mean(sample2):.3f}, std={np.std(sample2):.3f}, n={len(sample2)}")

    print("\n=== ПРОВЕРКА НОРМАЛЬНОСТИ ===")
    # Тест Шапиро-Уилка на нормальность
    _, p_norm1 = stats.shapiro(sample1)
    _, p_norm2 = stats.shapiro(sample2)
    print(f"Нормальность выборки 1: p={p_norm1:.9f}")
    print(f"Нормальность выборки 2: p={p_norm2:.9f}")

    normal_dist = (p_norm1 > alpha) and (p_norm2 > alpha)

    print("\n=== ПРОВЕРКА РАВЕНСТВА ДИСПЕРСИЙ ===")
    # Тест Левена на равенство дисперсий
    _, p_var = stats.levene(sample1, sample2)
    print(f"Равенство дисперсий: p={p_var:.4f}")

    equal_var = p_var > alpha

    print("\n=== СРАВНЕНИЕ СРЕДНИХ ===")

    if normal_dist:
        # Параметрические тесты
        if equal_var:
            t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=True)
            print(f"t-тест (равные дисперсии): p={p_val:.4f}")
        else:
            t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
            print(f"Welch t-тест (разные дисперсии): p={p_val:.4f}")
    else:
        print("Данные не нормальны, используем непараметрические тесты")

    # Непараметрические тесты (всегда)
    u_stat, p_mann = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
    print(f"Критерий Манна-Уитни: p={p_mann:.4f}")

    # Тест Колмогорова-Смирнова
    ks_stat, p_ks = stats.ks_2samp(sample1, sample2)
    print(f"Критерий Колмогорова-Смирнова: p={p_ks:.4f}")

    print(f"\n=== ЗАКЛЮЧЕНИЕ (α={alpha}) ===")
    if normal_dist and equal_var:
        conclusion = "отвергаем H0" if p_val < alpha else "не отвергаем H0"
        print(f"Основной тест (t-тест): {conclusion}")
    else:
        conclusion = "отвергаем H0" if p_mann < alpha else "не отвергаем H0"
        print(f"Основной тест (Манн-Уитни): {conclusion}")
def bar_metrics_draw(df, name, title, metric, met = 'mean'):
    fig, ax = plt.subplots(figsize=(20, 9))
    x = np.arange(len(df.index))
    width = 0.2
    colors = ['royalblue', 'indianred', 'limegreen']
    multiplier = 0

    alphs = {}
    for column in df.columns:
        alphs[column] = 0.4
    if met is not None:
        alphs[met] = 0.95
    for i, column in enumerate(df.columns):
        offset = width * multiplier
        bars = ax.bar(x + offset, df[column], width, label=column, color=colors[i], alpha=alphs[column])
        multiplier += 1
    # Настройка осей и подписей
    #ax.set_xlabel('Строки DataFrame')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x + width)  # центрируем подписи между группами полосок
    labs = []
    for ind in df.index:
        labs.append(underscore(ind))
    ax.set_xticklabels(labs)  # используем имена строк как подписи
    ax.legend()
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(name + '.png', dpi=300, bbox_inches='tight')

    plt.close()
    #plt.show()
def box_plot_metrics_draw(dic, xlabs, title, name, ref = None, p_values_dict = None, test_name = None):
    fig, ax = plt.subplots(figsize=(16, 9))
    bp = ax.boxplot(dic.values(), showmeans=True, meanline=True, medianprops=medianprops, meanprops=meanprops)
    #print(xlabs)
    ax.set_xticklabels(xlabs)
    legend_handles = [bp['medians'][0], bp['means'][0]]
    legend_labels = ['Медиана', 'Среднее']
    if ref is not None:
        ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        legend_handles.append(ax.lines[-1])
        legend_labels.append('Значение ' + ref)
    if p_values_dict is not None:
        p_text = '\n'.join([f'{key}: p={p_val:.3f}\n{test_name[key]}' for key, p_val in p_values_dict.items()])
        ax.text(0.02, 0.98, p_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.6))
        # for key, p_val in p_values_dict.items():
        #     invisible_line = ax.plot([], [], ' ')[0]  # невидимая линия
        #     legend_handles.append(invisible_line)
        #     legend_labels.append(f'{key}: p={p_val:.3f}')

    ax.legend(legend_handles, legend_labels, loc='upper right')
    ax.set_title(title)
    plt.savefig(name + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()


def underscore(s):
    pattern = r'_([^_]*_[^_]*_[^_]*)$'
    replacement = r'\n\1'

    return re.sub(pattern, replacement, s)
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
           'novelty': {},
           'weighted prec': {}}
    for user in rating[Columns.User].unique()[:100]:

        filename = f"{papka}metrics_user{user}.csv"
        if os.path.exists(filename) and user not in user_blacklist:
            df_dic[user] = pd.read_csv(filename)

    labs = []
    for p in ps:
        labs.append(lab_title_make(param_id, p)[0])
    #print(labs)
    title_part = lab_title_make(param_id)[1]
    for num, key in enumerate(dic.keys()):
        dic[key] = {}
        for p in ps:
            dic[key][p] = []
            #print(p)
            for combination in product(*param_values):
                #print(combination)
                #col_key = string_make(combination, param_id)
                #print(col_key)

                col = string_make(combination, param_id, p, False)
                #print(col)
                for user, df in df_dic.items():
                    if col in df.columns:
                        v = df.at[num, col]
                        #print(v)
                        dic[key][p].append(v)
        p_dic = {}
        test_name_dic = {}
        for i in range(1, len(ps)):
            (p_dic[str(labs[0]) + ' ~ ' + str(labs[i])],
             test_name_dic[str(labs[0]) + ' ~ ' + str(labs[i])]) = (
                mini_comparison(dic[key][ps[0]], dic[key][ps[i]], key))
            # dic[key][ps[i]] = np.array(dic[key][ps[i]]) - np.array(dic[key][ps[0]])
        name = papka + dic_params[param_id] + '_plots/' + key + '-' + dic_params[param_id]
        title = 'Значение ' + key + ' в зависимости от ' + title_part
        box_plot_metrics_draw(dic[key], labs, title, name, p_values_dict = p_dic, test_name = test_name_dic)
        p_dic = {}
        test_name_dic = {}
    plt.close('all')

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
    print(f"{dic_params[param_id]}: {ps}")
    dic = {'prec': {},
           'recall': {},
           'ndcg': {},
           'serendipity': {},
           'novelty': {},
           'weighted prec': {}}
    for user in rating[Columns.User].unique()[:100]:

        filename = f"{papka}metrics_user{user}.csv"
        if os.path.exists(filename) and user not in user_blacklist:
            df_dic[user] = pd.read_csv(filename)

    labs = []
    for p in ps:
        labs.append(lab_title_make(param_id, p)[0])

    title_part = lab_title_make(param_id)[1]
    for num, key in enumerate(dic.keys()):
        print(key)
        for combination in product(*param_values):
            skip_flag = False
            col_key = string_make(combination, param_id)
            #print(col_key)
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
            #fin_dic = deepcopy(dic)
            p_dic = {}
            test_name_dic = {}
            pflag = 0
            for i in range(1, len(ps)):
                (p_dic[str(labs[0]) + ' ~ ' + str(labs[i])],
                 test_name_dic[str(labs[0]) + ' ~ ' + str(labs[i])]) = (
                    mini_comparison(dic[key][col_key][ps[0]], dic[key][col_key][ps[i]], key))
                if pflag == 0 and p_dic[str(labs[0]) + ' ~ ' + str(labs[i])] <= 0.05:
                    pflag = 1
                #dic[key][col_key][ps[i]] = np.array(dic[key][col_key][ps[i]]) - np.array(dic[key][col_key][ps[0]])
            # dic[key][col_key].pop(ps[0], None)
            # name = papka + dic_params[param_id] + '_plots/diff/' + key + '-no ' + dic_params[param_id] + '=' + str(ps[0]) + '@' + col_key
            # title = key + ': Сравнение' + ' с ' + str(labs[0]) + ' (' + col_key + ')'
            # box_plot_metrics_draw(dic[key][col_key], labs[1:], title, name, str(labs[0]), p_dic, test_name_dic)
            if pflag == 1:
                name = papka + dic_params[param_id] + '_plots/' + key + '-' + dic_params[param_id] + '@' + col_key
                title = 'Значение ' + key + ' в зависимости от ' + title_part + ' (' + col_key + ')'
                box_plot_metrics_draw(dic[key][col_key], labs, title, name, p_values_dict = p_dic, test_name = test_name_dic)
        plt.close('all')
def metrics_draw_ml(inner_param_grid, alt = None):
    param_grid = deepcopy(inner_param_grid)
    #print(param_grid)
    df_dic = {}

    param_values = param_grid.values()

    dic = {'prec': {},
           'recall': {},
           'ndcg': {},
           'serendipity': {},
           'novelty': {},
           'weighted prec': {}}
    models_list = ['KNN TF-IDF', 'ALS', 'Random', 'Popular']
    for user in rating[Columns.User].unique()[:100]:
        #print(user)
        filename = f"{papka}metrics_user{user}.csv"
        filename_ml = f"{papka_ml}metrics_user{user}.csv"
        #print(filename, filename_ml)
        if os.path.exists(filename) and os.path.exists(filename_ml) and user not in user_blacklist:
            df_dic[user] = (pd.read_csv(filename), pd.read_csv(filename_ml))

    #print(df_dic)
    for num, key in enumerate(dic.keys()):
        print(key)
        model_metrics = {}
        for mod in models_list:
            user_list_ml = []
            for user, dfs in df_dic.items():
                user_list_ml.append(dfs[1].at[num, mod])
            model_metrics[mod] = user_list_ml
        for combination in product(*param_values):
            col = (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
                combination[3]) +
                          '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                        combination[5]))
            dic[key][col] = {}
            user_list = []
            for user, dfs in df_dic.items():
                if col in dfs[0].columns:
                    #print(dfs[0][col][num])
                    v = dfs[0].at[num, col]
                    # print(v)
                    user_list.append(v)

            dic[key][col][col] = user_list
            dic[key][col].update(model_metrics)

            p_dic = {}
            test_name_dic = {}
            pflag = 0
            for mod in models_list:
                #print(np.mean(dic[key][col][col]), dic[key][col][mod])
                (p_dic[str(col) + ' ~ ' + mod],
                 test_name_dic[str(col) + ' ~ ' + mod]) = (
                    mini_comparison(dic[key][col][col], dic[key][col][mod], key, alt = alt))
                #print(p_dic[str(col) + ' ~ ' + mod])
                if p_dic[str(col) + ' ~ ' + mod] <= 0.05 and np.mean(dic[key][col][col]) > np.mean(dic[key][col][mod]):
                    pflag += 1
                    #print(col, mod)
                #difference = np.array(dic[key][col][col]) - np.array(dic[key][col][mod])
                #name = papka_ml + 'plots/diff/' + key + '_' + mod +'_' + col
                #title = key + ': Сравнение' + ' с ' + mod
                #box_plot_metrics_draw({'f':difference}, [col], title, name, models_list[6],
                #                      {str(col) + ' ~ ' + mod:p_dic[str(col) + ' ~ ' + mod]},
                #                      {str(col) + ' ~ ' + mod:test_name_dic[str(col) + ' ~ ' + mod]})
            if pflag >= 1:
                name = papka_ml + 'plots/' + key + '-' + col
                title = 'Значение ' + key + ' (' + col + ')'
                box_plot_metrics_draw(dic[key][col], ['election'] + models_list, title, name, p_values_dict = p_dic, test_name = test_name_dic)
        plt.close('all')

def get_top_k(dataframe, k, metric_key, ascending=[False, False, True]):
    result = {}
    meh = 0
    for col in dataframe.columns:
        sorted_df = dataframe.sort_values(by=col, ascending=ascending[meh])
        meh += 1
        sorted_df.to_csv(papka + 'tops/top_' + metric_key + '_' + str(k) + '_by_' + col + '.csv')
        result[col] = sorted_df.head(k)
    return result
def top_draw(inner_param_grid, top_k = 20):
    param_grid = deepcopy(inner_param_grid)
    #print(param_grid)
    df_dic = {}

    param_values = param_grid.values()

    dic = {'prec': {},
           'recall': {},
           'ndcg': {},
           'serendipity': {},
           'novelty': {},
           'weighted prec': {}}
    for user in rating[Columns.User].unique()[:100]:

        filename = f"{papka}metrics_user{user}.csv"
        if os.path.exists(filename) and user not in user_blacklist:
            df_dic[user] = pd.read_csv(filename)

    for num, key in enumerate(dic.keys()):
        print(key)
        for combination in product(*param_values):
            col_key= (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
                combination[3]) +
                          '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                        combination[5]))
            #print(col_key)
            dic[key][col_key] = {}
            user_list = []
            for user, df in df_dic.items():
                if col_key in df.columns:
                    v = df.at[num, col_key]
                    user_list.append(v)
            dic[key][col_key] = (np.mean(user_list), np.median(user_list), np.std(user_list))

        df_stats = pd.DataFrame.from_dict(dic[key], orient='index',
                                    columns=['mean', 'median', 'std'])
        top_dic = get_top_k(df_stats, top_k, key)
        for met in ['mean', 'median', 'std']:
            name = papka + 'tops/' + key + '_' + met
            title = 'top ' + str(top_k) + ' ' + key + ' by ' + met
            bar_metrics_draw(top_dic[met], name, title, key)
        plt.close('all')
def top_draw_ml(inner_param_grid, top_k = 20):
    param_grid = deepcopy(inner_param_grid)
    #print(param_grid)
    df_dic = {}

    param_values = param_grid.values()

    dic = {'prec': {},
           'recall': {},
           'ndcg': {},
           'serendipity': {},
           'novelty': {},
           'weighted prec':{}}
    models_list = ['KNN TF-IDF', 'ALS', 'Random', 'Popular']

    for user in rating[Columns.User].unique()[:100]:

        filename = f"{papka}metrics_user{user}.csv"
        filename_ml = f"{papka_ml}metrics_user{user}.csv"
        if os.path.exists(filename) and os.path.exists(filename_ml) and user not in user_blacklist:
            df_dic[user] = (pd.read_csv(filename), pd.read_csv(filename_ml))

    for num, key in enumerate(dic.keys()):
        print(key)
        model_metrics = {}
        for mod in models_list:
            dic[key][mod] = {}
            user_list_ml = []
            for user, dfs in df_dic.items():
                user_list_ml.append(dfs[1].at[num, mod])
            model_metrics[mod] = user_list_ml
            dic[key][mod] = (np.mean(user_list_ml), np.median(user_list_ml), np.std(user_list_ml))
        for combination in product(*param_values):
            col_key= (combination[0] + '_' + combination[1] + '_deg=' + str(combination[2]) + '_size=' + str(
                combination[3]) +
                          '_weighted_' * combination[4] + '_antirec_' * (1 - combination[4]) + 'rate=' + str(
                        combination[5]))
            #print(col_key)
            dic[key][col_key] = {}
            user_list = []
            for user, dfs in df_dic.items():
                if col_key in dfs[0].columns:
                    v = dfs[0].at[num, col_key]
                    user_list.append(v)
            dic[key][col_key] = (np.mean(user_list), np.median(user_list), np.std(user_list))

        df_stats = pd.DataFrame.from_dict(dic[key], orient='index',
                                    columns=['mean', 'median', 'std'])
        top_dic = get_top_k(df_stats, top_k, key)
        for met in ['mean', 'median', 'std']:
            name = papka_ml + 'tops/' + key + '_' + met
            title = 'top ' + str(top_k) + ' ' + key + ' by ' + met
            bar_metrics_draw(top_dic[met], name, title, key, met = met)
        plt.close('all')
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
params_grid = {'rule':['SNTV', 'STV_star', 'STV_basic'],
               'dist_method':['jaccar', 'cosine', 'pearson', 'spearman', 'kendall'],
               'degrees':[7],
               'size':[10],
               'weighted':[True, False],
               'series_rate':[0, 1, 2, 3]}

# df_dic = {}
# for user in rating[Columns.User].unique()[:100]:
#
#     filename = f"my_films/test1/metrics_user{user}.csv"
#     if os.path.exists(filename):
#         df_dic[user] = pd.read_csv(filename)
# for num in range(6):
#     print(num)
#     user_list = []
#     for user, df in df_dic.items():
#         for col_key in df.columns[1:]:
#             v = df.at[num, col_key]
#             #print(v)
#             user_list.append(v)
#         print('user', user, 'mean', np.array(user_list).mean())
#     print('#' + str(num) + ': ' + str(np.array(user_list).mean()))

user_blacklist = set()
# user_blacklist_bad = {1, 2, 3, 11, 25, 29, 33, 43, 50, 51, 53, 55, 57, 64, 66, 67, 68}
# user_blacklist_ml = {8, 15, 32, 36, 37, 44, 56, 58}

#top_draw(params_grid, top_k = 7)
#top_draw_ml(params_grid, top_k = 10)
#metrics_draw_ml(params_grid, alt = 'two-sided')
#metrics_draw(4, params_grid)

name = papka + 'tops/top_novelty_10_by_std.csv'
df = pd.read_csv(name)
weighted = ([], [], [])
antirec = ([], [], [])
for index, row in df.iterrows():
    unnamed_value = row['Unnamed: 0']
    #print(unnamed_value)
    if 'weighted' in unnamed_value:
        weighted[2].append(row['std'])
        weighted[1].append(row['median'])
        weighted[0].append(row['mean'])
    else:
        antirec[2].append(row['std'])
        antirec[1].append(row['median'])
        antirec[0].append(row['mean'])

#weighted = np.array(weighted)
#antirec = np.array(antirec)
print('weighted std is', np.std(weighted[2]))
print('antirec std is', np.std(antirec[2]))

print('weighted median is', np.std(weighted[1]))
print('antirec median is', np.std(antirec[1]))

print('weighted mean is', np.std(weighted[0]))
print('antirec mean is', np.std(antirec[0]))
metrics_draw_small(1, params_grid)
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

