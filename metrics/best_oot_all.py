
# ###################### used to run perl script ##########################

import pandas as pd
import subprocess
import ast


# class config:
#     COINCO = False
#     main_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_6/'
#
#     if COINCO:
#         path = main_path + 'coinco_con_sent_sim/results.csv'
#         best_path = main_path + 'coinco_con_sent_sim/pred.best'
#         oot_path = main_path + 'coinco_con_sent_sim/pred.oot'
#         best_results_path = main_path + 'coinco_con_sent_sim/_results.best'
#         oot_results_path = main_path + 'coinco_con_sent_sim/_results.oot'
#         golden_file = 'coinco/gold'
#
#     else:
#         path = main_path + 'semeval_con_sent_sim/results.csv'
#         best_path = main_path + 'semeval_con_sent_sim/pred.best'
#         oot_path = main_path + 'semeval_con_sent_sim/pred.oot'
#         best_results_path = main_path + 'semeval_con_sent_sim/_results.best'
#         oot_results_path = main_path + 'semeval_con_sent_sim/_results.oot'
#         golden_file = 'semeval/gold'


def write_results_lex_best(filepath, change_word, id, proposed_list, limit=1):
    f = open(filepath, "a")
    proposed_list = proposed_list[:limit]

    proposed_word = ';'.join(proposed_list)
    proposed_word = proposed_word.strip()

    f.write(change_word + " " + id + " :: " + proposed_word + "\n")
    f.close()
    return


def write_results_lex_oot(filepath, change_word, id, proposed_list, limit=10):
    f = open(filepath, "a")

    proposed_list = proposed_list[:limit]

    proposed_word = ';'.join(proposed_list)
    proposed_word = proposed_word.strip()

    f.write(change_word + " " + id + " ::: " + proposed_word + "\n")
    f.close()
    return


def calculation_perl(golden_file, output_results_best, output_results_out, results_file_best,
                     results_file_out):
    command = "perl metrics/score.pl " + output_results_best + " " + golden_file + " -t best > " + results_file_best + " -v"

    subprocess.run(command, shell=True)

    command = "perl metrics/score.pl " + output_results_out + " " + golden_file + " -t oot > " + results_file_out + " -v"
    subprocess.run(command, shell=True)
    return


def read_gold(gold_path):
    gold_file = open(gold_path, 'r')
    change_words = []
    ids = []
    for gold_line in gold_file:
        segments = gold_line.split("::")
        segment = segments[0].strip().split(' ')
        if len(segment) > 1:
            change_words.append(segment[0])
            ids.append(segment[1])

    return change_words, ids


def read_file(path, best_path, oot_path, best_results_path, oot_results_path, golden_file):
    df = pd.read_csv(path)
    change_words, ids = read_gold(golden_file)
    proposed_pred = df['pred_substitutes'].tolist()

    # change_words, ids = change_words[:2], ids[:2]

    assert len(change_words) == len(proposed_pred)

    for i in range(len(change_words)):
        change_word = change_words[i]
        id = ids[i]
        proposed = proposed_pred[i]
        proposed = ast.literal_eval(proposed)

        write_results_lex_best(best_path, change_word, id, proposed, limit=1)
        write_results_lex_oot(oot_path, change_word, id, proposed, limit=10)

    calculation_perl(golden_file, best_path, oot_path, best_results_path, oot_results_path)


# read_file(config.path, config.best_path, config.oot_path, config.best_results_path, config.oot_results_path,
#          config.golden_file)


path_list = ['/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_0/',
             '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_1/',
             '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_2/',
             '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_3/',
             '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_4/',
             '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_5/']

for main_path in path_list:
    path = main_path + 'coinco_con_sent_sim/results.csv'
    best_path = main_path + 'coinco_con_sent_sim/pred.best'
    oot_path = main_path + 'coinco_con_sent_sim/pred.oot'
    best_results_path = main_path + 'coinco_con_sent_sim/_results.best'
    oot_results_path = main_path + 'coinco_con_sent_sim/_results.oot'
    golden_file = 'coinco/gold'

    calculation_perl(golden_file, best_path, oot_path, best_results_path,
                 oot_results_path)


for main_path in path_list:
    path = main_path + 'semeval_con_sent_sim/results.csv'
    best_path = main_path + 'semeval_con_sent_sim/pred.best'
    oot_path = main_path + 'semeval_con_sent_sim/pred.oot'
    best_results_path = main_path + 'semeval_con_sent_sim/_results.best'
    oot_results_path = main_path + 'semeval_con_sent_sim/_results.oot'
    golden_file = 'semeval/gold'

    calculation_perl(golden_file, best_path, oot_path, best_results_path,
                 oot_results_path)