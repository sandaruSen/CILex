import pandas as pd
import subprocess
import ast
import tempfile
import numpy as np

class config:
    COINCO = False
    main_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_1/'

    if COINCO:
        path = main_path + 'coinco_con_sent_sim/results.csv'
        best_path = main_path + 'coinco_con_sent_sim/pred.best'
        oot_path = main_path + 'coinco_con_sent_sim/pred.oot'
        best_results_path = main_path + 'coinco_con_sent_sim/results.best'
        oot_results_path = main_path + 'coinco_con_sent_sim/results.oot'
        golden_file = 'coinco/gold'

        best_results_all = main_path + 'coinco_con_sent_sim/pred_all.best'
        oot_results_all = main_path + 'coinco_con_sent_sim/pred_all.oot'

        variance_path = main_path + 'coinco_con_sent_sim/var.txt'

    else:
        path = main_path + 'semeval_con_sent_sim/results.csv'
        best_path = main_path + 'semeval_con_sent_sim/pred.best'
        oot_path = main_path + 'semeval_con_sent_sim/pred.oot'
        best_results_path = main_path + 'semeval_con_sent_sim/results.best'
        oot_results_path = main_path + 'semeval_con_sent_sim/results.oot'
        golden_file = 'semeval/gold'


        best_results_all = main_path + 'semeval_con_sent_sim/pred_all.best'
        oot_results_all = main_path + 'semeval_con_sent_sim/pred_all.oot'

        variance_path = main_path + 'semeval_con_sent_sim/var.txt'



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
    command = "perl metrics/score.pl " + output_results_best + " " + golden_file + " -t best > " + results_file_best  # + " -v"

    subprocess.run(command, shell=True)

    command = "perl metrics/score.pl " + output_results_out + " " + golden_file + " -t oot > " + results_file_out  # + " -v"
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


def compute_variance(best_results_all, oot_results_all, variance_path):
    with open(best_results_all, "r") as best_data:
        best_lines = best_data.readlines()
    with open(oot_results_all, "r") as oot_data:
        oot_lines = oot_data.readlines()

    var_best = []
    var_best_mode = []
    for i in best_lines:
        item = i.split(' ')
        var_best.append(float(item[0]))
        var_best_mode.append(float(item[2]))

    var_oot = []
    var_oot_mode = []
    for i in oot_lines:
        item = i.split(' ')
        var_oot.append(float(item[0]))
        var_oot_mode.append(float(item[2]))


    best_var = np.var(var_best)
    best_mode_var = np.var(var_best_mode)
    oot_var = np.var(var_oot)
    oot_mode_var = np.var(var_oot_mode)

    f = open(variance_path, "w")
    f.write("best variance: " + repr(best_var) +"\n")
    f.write("best mode variance: " + repr(best_mode_var) +"\n")
    f.write("oot variance: " + repr(oot_var) +"\n")
    f.write("oot mode variance: " + repr(oot_mode_var) +"\n")
    f.close()


def read_file_(best_path, oot_path, golden_file):
    with open(best_path, "r") as best_data:
        best_lines = best_data.readlines()
    with open(oot_path, "r") as oot_data:
        oot_lines = oot_data.readlines()

    assert len(best_lines) == len(oot_lines)
    for i in range(len(best_lines)):
        temp_best = tempfile.TemporaryFile()
        temp_oot = tempfile.TemporaryFile()
        try:
            temp_best.write(best_lines[i])
            temp_oot.write(oot_lines[i])
            print(temp_best.read())
            calculation_perl(golden_file, temp_best, temp_oot, config.best_results_all, config.oot_results_all)

        finally:
            temp_best.close()
            temp_oot.close()


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

    with open(best_path, "r") as best_data:
        best_lines = best_data.readlines()
    with open(oot_path, "r") as oot_data:
        oot_lines = oot_data.readlines()

    assert len(best_lines) == len(oot_lines)
    for i in range(len(best_lines)):
        temp_best = tempfile.TemporaryFile()
        temp_oot = tempfile.TemporaryFile()
        try:
            temp_best.write(best_lines[i])
            temp_oot.write(oot_lines[i])
            print(temp_best.read())
            calculation_perl(golden_file, temp_best, temp_oot, config.best_results_all, config.oot_results_all)

        finally:
            temp_best.close()

    # calculation_perl(golden_file, best_path, oot_path, best_results_path, oot_results_path)


# read_file(config.path, config.best_path, config.oot_path, config.best_results_path, config.oot_results_path,
#           config.golden_file)


read_file_(config.best_path, config.oot_path, config.golden_file)

compute_variance(config.best_results_all, config.oot_results_all, config.variance_path)
