# Example of the Wilcoxon Signed-Rank Test

import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import pearsonr

main_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/'
path1 = main_path + 'best_0/' + 'coinco_con_sent_sim/results.csv'
path2 = main_path + 'best_1/' + 'coinco_con_sent_sim/results.csv'


# path1 = main_path + 'best_0/' + 'semeval_con_sent_sim/results.csv'
# path2 = main_path + 'best_1/' + 'semeval_con_sent_sim/results.csv'


def wilcoxon_test(data1, data2):
    stat, p = wilcoxon(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

def pearson_test(data1, data2):
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)


def read_data(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    prec3_1 = df1['prec@3'].tolist()
    prec3_2 = df2['prec@3'].tolist()

    prec1_1 = df1['prec@1'].tolist()
    prec1_2 = df2['prec@1'].tolist()

    return prec1_1, prec1_2, prec3_1, prec3_2



prec1_1, prec1_2, prec3_1, prec3_2 = read_data(path1, path2)

wilcoxon_test(prec1_1, prec1_2)
wilcoxon_test(prec3_1, prec3_2)





pearson_test(prec1_1, prec1_2)
pearson_test(prec3_1, prec3_2)