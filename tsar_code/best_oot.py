import pandas as pd
import subprocess
import ast

#
# def clean_proposed(proposed):
#     proposed_temp = {}
#     for word in proposed:
#         word_temp = word.replace("_", " ")
#         word_temp = word_temp.replace("-", " ")
#         if word_temp not in proposed_temp:
#             proposed_temp[word_temp] = proposed[word]
#     return proposed_temp


class config:
    COINCO = False

    if COINCO:
        path = 'coinco/results.csv'
        best_path = 'coinco/pred.best'
        oot_path = 'coinco/pred.oot'
        best_results_path = 'coinco/results.best'
        oot_results_path = 'coinco/results.oot'
        golden_file = 'coinco/gold'

    else:
        path = 'semeval/results.csv'
        best_path = 'semeval/pred.best'
        oot_path = 'semeval/pred.oot'
        best_results_path = 'semeval/results.best'
        oot_results_path = 'semeval/results.oot'
        golden_file = 'semeval/gold'


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
        if len(segment)>1:
           change_words.append(segment[0])
           ids.append(segment[1])

    return change_words, ids


def read_file(path, best_path, oot_path, best_results_path, oot_results_path, golden_file):
    df = pd.read_csv(path)
    change_words, ids = read_gold(golden_file)
    proposed_pred = df['pred_substitutes'].tolist()

    #change_words, ids = change_words[:2], ids[:2]

    assert len(change_words) == len(proposed_pred)

    for i in range(len(change_words)):
        change_word = change_words[i]
        id = ids[i]
        proposed = proposed_pred[i]
        proposed = ast.literal_eval(proposed)

        write_results_lex_best(best_path, change_word, id, proposed, limit=1)
        write_results_lex_oot(oot_path, change_word, id, proposed, limit=10)

    calculation_perl(golden_file, best_path, oot_path, best_results_path, oot_results_path)


read_file(config.path, config.best_path, config.oot_path, config.best_results_path, config.oot_results_path,
          config.golden_file)
