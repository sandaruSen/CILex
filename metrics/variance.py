import numpy as np
import pandas as pd


def compute_variance(best_results_all, oot_results_all, variance_path, results_path):
    with open(best_results_all, "r") as best_data:
        _best_lines = best_data.readlines()[:-4]
    with open(oot_results_all, "r") as oot_data:
        _oot_lines = oot_data.readlines()[:-4]

    best_lines = []
    best_credits = []
    best_response = []
    best_score = []
    best_ids = []
    best_mode_cre = []
    best_mode_res = []
    best_mode_sco = []
    for line in _best_lines:
        if 'mode' not in line:
            best_lines.append(line)
            line_list = line.split(' ')
            best_credits.append(float(line_list[4]))
            best_score.append(float(line_list[-1]))
            best_response.append(float(line_list[9].split(':')[0]))

        line_list = line.split(' ')
        if 'mode' in line:
            best_ids.append(int(line_list[2]))

        if 'mode' not in line and int(line_list[2]) in best_ids:
            best_mode_cre.append(float(line_list[4]))
            best_mode_res.append(float(line_list[-1]))
            best_mode_sco.append(float(line_list[9].split(':')[0]))

    # print(sum(best_mode_cre) / len(best_mode_cre))
    # print(sum(best_mode_res) / len(best_mode_res))
    # print(sum(best_mode_sco) / len(best_mode_sco))
    #
    # print(sum(best_credits) / len(best_credits))
    # print(sum(best_score) / len(best_score))
    # print(sum(best_response) / len(best_response))

    oot_lines = []
    oot_credits = []
    oot_response = []
    oot_score = []

    for line in _oot_lines:
        if 'mode' not in line and 'Total =' not in line and 'precision =' not in line:
            oot_lines.append(line)
            line_list = line.split(' ')
            oot_credits.append(float(line_list[4]))
            oot_score.append(float(line_list[-1]))
            oot_response.append(float(line_list[7].split(':')[0]))

    print(sum(oot_credits) / len(oot_credits))
    print(sum(oot_score) / len(oot_score))
    print(sum(oot_response) / len(oot_response))

    df = pd.read_csv(results_path)
    prec1 = df['prec@1'].tolist()
    prec3 = df['prec@3'].tolist()
    gap_normalized = df['gap_normalized'].tolist()
    gap = df['gap'].tolist()

    best_var = np.var(best_credits)
    oot_var = np.var(oot_credits)
    prec1_var = np.var(prec1)
    prec3_var = np.var(prec3)
    gap_normalized_var = np.var(gap_normalized)
    gap_var = np.var(gap)

    # best_mode_var = np.var(var_best_mode)
    # oot_mode_var = np.var(var_oot_mode)

    f = open(variance_path, "w")
    f.write("best variance: " + repr(best_var) + "\n")
    f.write("oot variance: " + repr(oot_var) + "\n")
    f.write("p@1 variance: " + repr(prec1_var) + "\n")
    f.write("p@3 variance: " + repr(prec3_var) + "\n")
    f.write("gap norm variance: " + repr(gap_normalized_var) + "\n")
    f.write("gap variance: " + repr(gap_var) + "\n")

    # print(sum(gap)/len(gap))
    # print(sum(gap_normalized)/len(gap_normalized))

    # f.write("best mode variance: " + repr(best_mode_var) +"\n")
    # f.write("oot mode variance: " + repr(oot_mode_var) +"\n")

    f.close()


def prec_cal(variance_path, results_path):
    df = pd.read_csv(results_path)
    prec1 = df['prec@1'].tolist()
    prec3 = df['prec@3'].tolist()

    count_0 = (prec1.count(0) / len(prec1)) * 100
    count_1 = (prec1.count(1) / len(prec1)) * 100

    count_0_pre3 = (prec3.count(0) / len(prec3)) * 100
    count_0_3_pre3 = (prec3.count(0.3333333333333333) / len(prec3)) * 100
    count_0_6_pre3 = (prec3.count(0.6666666666666666) / len(prec3)) * 100
    count_1_pre3 = (prec3.count(1) / len(prec3)) * 100

    f = open(variance_path, "a")
    f.write("prec3 - 0: " + repr(count_0_pre3) + "\n")
    f.write("prec3 - 0.3: " + repr(count_0_3_pre3) + "\n")
    f.write("prec3 - 0.6: " + repr(count_0_6_pre3) + "\n")
    f.write("prec3 - 1: " + repr(count_1_pre3) + "\n")


all_results_path = 'analysis/results.csv'
best_results_path = 'analysis/_results.best'
oot_results_path = 'analysis/_results.oot'
variance_path = 'analysis/_results.var'
# compute_variance(best_results_path, oot_results_path, variance_path, all_results_path)
prec_cal(variance_path, all_results_path)

# path_list = ['/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_0/',
#              '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_1/',
#              '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_2/',
#              '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_3/',
#              '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_4/',
#              '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_5/']
#
# for main_path in path_list:
#     best_results_path = main_path + 'coinco_con_sent_sim/_results.best'
#     oot_results_path = main_path + 'coinco_con_sent_sim/_results.oot'
#     variance_path = main_path + 'coinco_con_sent_sim/_results.var'
#     all_results_path =  main_path + 'coinco_con_sent_sim/results.csv'
#     compute_variance(best_results_path, oot_results_path, variance_path, all_results_path)
#
#
# for main_path in path_list:
#     best_results_path = main_path + 'semeval_con_sent_sim/_results.best'
#     oot_results_path = main_path + 'semeval_con_sent_sim/_results.oot'
#     variance_path = main_path + 'semeval_con_sent_sim/_results.var'
#     all_results_path =  main_path + 'semeval_con_sent_sim/results.csv'
#     compute_variance(best_results_path, oot_results_path, variance_path, all_results_path)
