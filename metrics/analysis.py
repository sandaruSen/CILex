import pandas as pd
from lexsubgen.lexsubcon.wordnet import Wordnet
import ast

wordnet_gloss = Wordnet()


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


def created_proposed_list(change_word, wordnet_gloss, pos_tag):
    gloss_list, synset, _ = wordnet_gloss.getSenses(change_word, pos_tag)
    synonyms = {}
    synonyms_final = {}

    for syn in synset:
        # adding lemmas
        for l in syn.lemmas():
            synonyms[l.name().lower()] = 0
        # adding hypernyms
        for syn_temp in syn.hypernyms():
            for l in syn_temp.lemmas():
                synonyms[l.name().lower()] = 0
        # adding hyponyms
        for syn_temp in syn.hyponyms():
            for l in syn_temp.lemmas():
                synonyms[l.name().lower()] = 0
    try:
        del synonyms[change_word]
    except:
        pass

    for word in synonyms:
        word_temp = word.replace("_", " ")
        word_temp = word_temp.replace("-", " ")
        word_temp = word_temp.replace("'", "")
        synonyms_final[word_temp] = 0

    return list(synonyms_final.keys())


def analysis(output_path, filtered_df_path, results_path):
    df = pd.read_csv(results_path)
    prec3 = df['prec@3'].tolist()

    count_0_pre3 = prec3.count(0)
    count_0_3_pre3 = prec3.count(0.3333333333333333)
    count_0_6_pre3 = prec3.count(0.6666666666666666)
    count_1_pre3 = prec3.count(1)

    filtered_df = df.loc[df['prec@3'] == 0]
    filtered_df.to_csv(filtered_df_path)

    print(len(filtered_df))
    print(count_0_pre3)

    pred_in_wordnet = []

    # print(filtered_df.loc[df['target_pos_tag'] =='n.v'])

    not_in_wordnet = []

    for id, row in filtered_df.iterrows():

        pred = ast.literal_eval(row['pred_substitutes'])[:3]
        context = ' '.join(ast.literal_eval(row['context']))
        target_word = row['target_word']
        gold_substitutes = ast.literal_eval(row['gold_substitutes'])

        target_lemma = row['target_lemma']
        target_pos_tag = row['target_pos_tag']

        if target_pos_tag == 'n.v':
            target_pos_tag = 'v'
        if target_pos_tag == 'a.n':
            target_pos_tag = 'n'
        if not pd.isna(target_lemma):
            wordnet_words = created_proposed_list(target_lemma, wordnet_gloss, target_pos_tag.lower())

            num = 0
            for sub in pred:
                if sub in wordnet_words:
                    num += 1

            pred_in_wordnet.append(num)

            if num == 0:
                not_in_wordnet.append([context, target_word, target_lemma, gold_substitutes, row['pred_substitutes']])

    count0 = pred_in_wordnet.count(0)
    count1 = pred_in_wordnet.count(1)
    count2 = pred_in_wordnet.count(2)
    count3 = pred_in_wordnet.count(3)

    f = open(output_path, "w")
    f.write("length of all the data: " + repr(len(df)) + "\n\n")

    f.write("0 from wordnet: " + repr(count0) + "\n")
    f.write("1 from wordnet: " + repr(count1) + "\n")
    f.write("2 from wordnet: " + repr(count2) + "\n")
    f.write("3 from wordnet: " + repr(count3) + "\n\n")

    count0_ = (count0 / len(df)) * 100
    count1_ = (count1 / len(df)) * 100
    count2_ = (count2 / len(df)) * 100
    count3_ = (count3 / len(df)) * 100

    f.write("0 from wordnet: % based on all data: " + repr(count0_) + "\n")
    f.write("1 from wordnet: % based on all data: " + repr(count1_) + "\n")
    f.write("2 from wordnet: % based on all data: " + repr(count2_) + "\n")
    f.write("3 from wordnet: % based on all data: " + repr(count3_) + "\n\n")

    count0__ = (count0 / len(filtered_df)) * 100
    count1__ = (count1 / len(filtered_df)) * 100
    count2__ = (count2 / len(filtered_df)) * 100
    count3__ = (count3 / len(filtered_df)) * 100

    f.write("0 from wordnet: % based on filtered 0 data: " + repr(count0__) + "\n")
    f.write("1 from wordnet: % based on filtered 0 data: " + repr(count1__) + "\n")
    f.write("2 from wordnet: % based on filtered 0 data: " + repr(count2__) + "\n")
    f.write("3 from wordnet: % based on filtered 0 data: " + repr(count3__) + "\n\n")


all_results_path = 'semeval/results.csv'
output_path = 'semeval/zero_pred.txt'
filtered_df_path = 'semeval/zero_pred.csv'
analysis(output_path, filtered_df_path, all_results_path)


# prec_cal('results.var','results.csv')



# main_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_0/coinco_con_sent_sim/'
# all_results_path = main_path + 'results.csv'
# output_path = main_path + 'zero_pred.txt'
# filtered_df_path = main_path + 'zero_pred.csv'
# analysis(output_path, filtered_df_path, all_results_path)
#
# main_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/best_0/semeval_con_sent_sim/'
# all_results_path = main_path + 'results.csv'
# output_path = main_path + 'zero_pred.txt'
# filtered_df_path = main_path + 'zero_pred.csv'
# analysis(output_path, filtered_df_path, all_results_path)
