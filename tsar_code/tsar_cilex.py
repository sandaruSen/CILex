#for gpu experiments

import logging
from pathlib import Path
from typing import Dict, List, Any, NoReturn, Optional

import pandas as pd
from overrides import overrides
import copy
import csv

from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task

from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from lexsubgen.utils.file import dump_json
from lexsubgen.utils.params import read_config
from lexsubgen.lexsubcon.gloss_score_text import CGLOSS
from lexsubgen.lexsubcon.validation_score import ValidationScore

from lexsubgen.lexsubcon.noise import Gloss_noise
from lexsubgen.lexsubcon.wordnet import Wordnet
from lexsubgen.lexsubcon.score import Cmasked
from lexsubgen.lexsubcon.similarity_new_predict import Similarity_new

from sentence_transformers import util

logger = logging.getLogger(Path(__file__).name)

DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent.parent / "debug" / Path(__file__).stem


class LexSubEvaluation(Task):
    def __init__(
            self,
            substitute_generator: SubstituteGenerator = None,
            dataset_reader: DatasetReader = None,
            verbose: bool = True,
            k_list: List[int] = (1, 3, 10),
            batch_size: int = 64,
            save_instance_results: bool = True,
            save_wordnet_relations: bool = False,
            save_target_rank: bool = False,
    ):
        """
        Main class for performing Lexical Substitution task evaluation.
        This evaluation computes metrics for two subtasks in Lexical Substitution task:

        - Candidate-ranking task (GAP, GAP_normalized, GAP_vocab_normalized).
        - All-word-ranking task (Precision@k, Recall@k, F1@k for k-best substitutes).

        Args:
            substitute_generator: Object that generate possible substitutes.
            dataset_reader: Object that can read dataset for Lexical Substitution task.
            verbose: Bool flag for verbosity.
            k_list: List of integer numbers for metrics. For example, if 'k_list' equal to [1, 3, 5],
                then there will calculating the following metrics:
                    - Precion@1, Recall@1, F1-score@1
                    - Precion@3, Recall@3, F1-score@3
                    - Precion@5, Recall@5, F1-score@5
            batch_size: Number of samples in batch for substitute generator.
        """
        super(LexSubEvaluation, self).__init__(
            substitute_generator=substitute_generator,
            dataset_reader=dataset_reader,
            verbose=verbose,
        )
        self.batch_size = batch_size
        self.k_list = k_list
        self.save_wordnet_relations = save_wordnet_relations
        self.save_target_rank = save_target_rank
        self.save_instance_results = save_instance_results

        self.gap_metrics = ["gap", "gap_normalized", "gap_vocab_normalized"]
        self.base_metrics = ["precision", "recall", "f1_score"]
        k_metrics = []
        for k in self.k_list:
            k_metrics.extend([f"prec@{k}", f"rec@{k}", f"f1@{k}"])
        self.metrics = self.gap_metrics + self.base_metrics + k_metrics

        # lex sub con specific details
        # for proposal score
        self.noise_gloss = Gloss_noise()
        self.wordnet_gloss = Wordnet()
        self.noise_type = "GLOSS"

        self.max_seq_length = 256
        self.lower_case = False
        self.proposal = Cmasked(self.max_seq_length, self.lower_case, pre_trained="bert-large-uncased")
        self.proposal.get_possible_words()
        self.proposal_flag = False

        # for sentence similarity score
        # self.train_file = ''
        # self.train_golden_file = ''
        # self.similarity_path_save = '../../../lexsubgen/lexsubcon/checkpoint/similarity_new_bert/'
        # self.similirity_sentence_new = Similarity_new(self.train_file, self.train_golden_file)
        # # self.similirity_sentence_new.load_model(model=self.similarity_path_save)
        # self.similirity_sentence_new.load_model()

        self.bita = 1
        self.gamma = 0.5
        self.delta = 0.05

        max_seq_length_gloss = 256
        do_lower_case = False
        max_seq_length = 128

        self.finding_gloss = CGLOSS('bert-base-uncased', max_seq_length_gloss, do_lower_case)

        self.validation = ValidationScore(max_seq_length, do_lower_case, pre_trained="bert-large-uncased")

        self.similirity_sentence_new = Similarity_new('', '')
        self.similirity_sentence_new.load_model()

    def get_sentence_Sim_Scores(self, tokens_lists, target_ids, proposed_words_list, given_words):
        text_similarities = []
        tokens = copy.deepcopy(tokens_lists)
        for i in range(len(tokens)):
            text = tokens[i]
            original_text = " ".join(text)
            index_word = target_ids[i]
            target_word = text[index_word]
            main_word = given_words[i]
            proposed_words = []
            synonyms = proposed_words_list[i]
            # for word in synonyms:
            #     proposed_words[word] = 0

            self.similirity_sentence_new.post_tag_target(main_word)
            for word in synonyms:
                similarity_score = self.similirity_sentence_new.initial_test(word, target_word)
                if similarity_score != 0:
                    list_temp = text
                    list_temp[int(index_word)] = word
                    text_update = " ".join(list_temp)
                    similarity_score = self.similirity_sentence_new.calculate_similarity_score(original_text,
                                                                                               text_update)
                proposed_words.append(self.bita * similarity_score)

            text_similarities.append(proposed_words)

        return text_similarities

    def get_gloss_sim_Scores(self, tokens_lists, target_ids, proposed_words_list, given_words):

        gloss_scores = []
        tokens = copy.deepcopy(tokens_lists)
        for i in range(len(tokens)):
            score_list = []

            original_text = tokens[i]
            text = " ".join(original_text)
            index_word = target_ids[i]
            change_word = tokens_lists[i][int(index_word)]
            main_word = given_words[i]
            proposed_words = proposed_words_list[i]

            self.finding_gloss.main_gloss_embedding = None
            if main_word.split('.')[0] == "":
                word_temp = "."
            else:
                word_temp = main_word.split('.')[0]

            gloss_list, _, lemmas = self.wordnet_gloss.getSenses(word_temp, main_word.split('.')[-1])

            # should be true
            if True:
                list_temp = text.split(" ")
                list_temp[int(index_word)] = '"' + list_temp[int(index_word)] + '"'
                text_for_gloss = " ".join(list_temp)
                for gloss in range(0, len(gloss_list)):
                    gloss_list[gloss] = change_word + " : " + gloss_list[gloss]

            if len(gloss_list) == 0:
                score_list = [0] * len(proposed_words)
                gloss_scores.append(score_list)


            else:
                gloss_embedding, _ = self.finding_gloss.find_best_gloss(text_for_gloss, gloss_list, main_word,
                                                                        lemmas,
                                                                        candidate=False, top_k=100)
                self.finding_gloss.save_main_gloss_embedding(gloss_embedding)

                for word in proposed_words:

                    gloss_list, _, lemmas = self.wordnet_gloss.getSenses(word)
                    if True:
                        list_temp = text.split(" ")
                        list_temp[int(index_word)] = '"' + word + '"'
                        text_for_gloss = " ".join(list_temp)
                        for gloss in range(0, len(gloss_list)):
                            gloss_list[gloss] = word + " : " + gloss_list[gloss]

                    if len(gloss_list) > 0:
                        _, similarity = self.finding_gloss.find_best_gloss(text_for_gloss, gloss_list, word, lemmas,
                                                                           candidate=True, top_k=100)
                        # proposed_words[word] = proposed_words[word] + self.delta * similarity

                        score_list.append(self.delta * similarity)

                    else:
                        similarity = 0
                        score_list.append(self.delta * similarity)

                gloss_scores.append(score_list)

        return gloss_scores

    def get_validation_score(self, tokens_lists, target_ids, proposed_words_list, given_words):

        validation_scores = []

        tokens = copy.deepcopy(tokens_lists)

        for i in range(len(tokens)):

            original_text = tokens[i]
            text = " ".join(original_text)
            index_word = target_ids[i]
            change_word = tokens_lists[i][int(index_word)]
            main_word = given_words[i]
            proposed_words = proposed_words_list[i]

            self.validation.get_contextual_weights_original(text, change_word, index_word, main_word)
            score_list = []
            for word in proposed_words:
                text_list = text.split(" ")
                text_list[int(index_word)] = word
                text_update = " ".join(text_list)
                self.validation.get_contextual_weights_update(text_update, word, int(index_word), main_word)
                similarity = self.validation.get_val_score(word)
                # proposed_words[word] = proposed_words[word] + self.gamma * similarity

                score_list.append(self.gamma * similarity)

            validation_scores.append(score_list)

        return validation_scores

    def query_answer(self, query, docs):
        model = self.finding_gloss

        # Encode query and documents
        query_emb = model.get_gloss_embedding(query)
        doc_emb = [model.get_gloss_embedding(doc) for doc in docs]
        scores = [util.pytorch_cos_sim(query_emb, doc)[0][0].item() for doc in doc_emb]

        # Compute dot score between query and all document embeddings
        # scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

        # Combine docs & scores
        doc_score_pairs = list(zip(docs, scores))

        # Sort by decreasing score
        doc_score_pairs_sorted = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Output passages & scores
        # print("Query:", query)
        # for doc, score in doc_score_pairs:
        #     print(score, doc)

        # get the definition with the maximum similarity score
        if len(doc_score_pairs) > 0:
            definition, score = doc_score_pairs_sorted[0]
            # get which index the definition with highest similarity score is at from all the definitions
            id = docs.index(definition)
        else:
            id = 0
            definition = query

        return id, definition, doc_score_pairs

    def get_wordnet_score(self, tokens_lists, target_ids, proposed_words_list, given_words):

        wordnet_scores = []
        tokens = copy.deepcopy(tokens_lists)
        for i in range(len(tokens)):
            score_list = []

            original_text = tokens[i]
            text = " ".join(original_text)
            index_word = target_ids[i]
            change_word = tokens_lists[i][int(index_word)]
            main_word = given_words[i]
            proposed_words = proposed_words_list[i]

            if main_word.split('.')[0] == "":
                word_temp = "."
            else:
                word_temp = main_word.split('.')[0]

            gloss_list, _, lemmas = self.wordnet_gloss.getSenses(word_temp, main_word.split('.')[-1])

            id, definition, doc_score_pairs = self.query_answer(text, gloss_list)

            updated_sent = [text.replace(word_temp, word) for word in proposed_words]
            # id, definition = self.query_answer(text, gloss_list)

            id, definition, doc_score_pairs = self.query_answer(definition, updated_sent)

            for doc, score in doc_score_pairs:
                score_list.append(self.delta * score)

            wordnet_scores.append(score_list)

        return wordnet_scores

    def combine_xlnet_sent(self, substitues, prob_values, sent_similarity_values):
        final_list = []
        for i in range(len(substitues)):
            item = []
            sub = substitues[i]
            prob = prob_values[i]
            sim = sent_similarity_values[i]
            for j in range(len(sub)):
                item.append([sub[j], 0.05 * prob[j] + sim[j]])

            item.sort(key=lambda x: x[1], reverse=True)
            final_list.append([i[0] for i in item])

        return final_list

    def combine_xlnet_sent_valid_gw(self, substitues, prob_values, sent_similarity_values, validation_similarities,
                                    wordnet):
        final_list = []
        for i in range(len(substitues)):
            item = []
            sub = substitues[i]
            prob = prob_values[i]
            sim = sent_similarity_values[i]
            val = validation_similarities[i]
            gloss = wordnet[i]
            for j in range(len(sub)):
                item.append([sub[j], 0.05 * prob[j] + 1 * sim[j] + val[j] + gloss[j]])

            item.sort(key=lambda x: x[1], reverse=True)
            final_list.append([i[0] for i in item])

        return final_list

    @overrides
    def get_metrics(self, dataset: pd.DataFrame):
        progress_bar = BatchReader(
            dataset["context"].tolist(),
            dataset["target_position"].tolist(),
            dataset["pos_tag"].tolist(),
            dataset["given_word"].tolist(),
            batch_size=self.batch_size,
        )

        dataset_sent = dataset.copy()
        dataset_gloss = dataset.copy()
        dataset_wordnet = dataset.copy()

        add_sent = []
        add_sent_gloss_valid = []
        add_sent_wordnet_valid = []
        for (
                tokens_lists,
                target_ids,
                pos_tags,
                given_words,
        ) in progress_bar:
            # Computing probability distribution over possible substitutes
            probs, word2id = self.substitute_generator.get_probs(
                tokens_lists, target_ids, pos_tags
            )

            # Selecting most probable substitutes from the obtained distribution
            pred_substitutes, prob_values = self.substitute_generator.substitutes_from_probs(
                probs, word2id, tokens_lists, target_ids
            )

            # ------------------------------

            sent_similarities = self.get_sentence_Sim_Scores(tokens_lists, target_ids, pred_substitutes, given_words)
            gloss_similarities = self.get_gloss_sim_Scores(tokens_lists, target_ids, pred_substitutes, given_words)
            wordnet_similarities = self.get_wordnet_score(tokens_lists, target_ids, pred_substitutes, given_words)
            validation_similarities = self.get_validation_score(tokens_lists, target_ids, pred_substitutes, given_words)

            sent_pred_substitutes = self.combine_xlnet_sent(pred_substitutes, prob_values, sent_similarities)
            sent_gloss_valid_substitutes = self.combine_xlnet_sent_valid_gw(pred_substitutes, prob_values,
                                                                            sent_similarities, validation_similarities,
                                                                            gloss_similarities)
            sent_wordnet_valid_substitutes = self.combine_xlnet_sent_valid_gw(pred_substitutes, prob_values,
                                                                              sent_similarities,
                                                                              validation_similarities,
                                                                              wordnet_similarities)

            sent_pred_substitutes = [i[:10] for i in sent_pred_substitutes]
            add_sent.extend(sent_pred_substitutes)

            sent_gloss_valid_substitutes = [i[:10] for i in sent_gloss_valid_substitutes]
            add_sent_gloss_valid.extend(sent_gloss_valid_substitutes)

            sent_wordnet_valid_substitutes = [i[:10] for i in sent_wordnet_valid_substitutes]
            add_sent_wordnet_valid.extend(sent_wordnet_valid_substitutes)


        for j in range(10):
            temp = []
            for i in add_sent:
                temp.append(i[j])
            dataset_sent[str(j)] = temp
        dataset_sent.drop(['context', 'target_position', 'pos_tag', 'given_word'], axis=1, inplace=True)
        print(dataset_sent)
        dataset_sent.to_csv('../subs/LexSubGen/configs/subst_generators/tsar/sent.tsv', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)


        for j in range(10):
            temp = []
            for i in add_sent_gloss_valid:
                temp.append(i[j])
            dataset_gloss[str(j)] = temp
        dataset_gloss.drop(['context', 'target_position', 'pos_tag', 'given_word'], axis=1, inplace=True)
        dataset_gloss.to_csv('../subs/LexSubGen/configs/subst_generators/tsar/sent_gloss.tsv', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)


        for j in range(10):
            temp = []
            for i in add_sent_wordnet_valid:
                temp.append(i[j])
            dataset_wordnet[str(j)] = temp
        dataset_wordnet.drop(['context', 'target_position', 'pos_tag', 'given_word'], axis=1, inplace=True)
        dataset_wordnet.to_csv('../subs/LexSubGen/configs/subst_generators/tsar/sent_wordnet.tsv', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)



    def solve(
            self,
            substgen_config_path: str,
            dataset_config_path: str,
            run_dir: str = DEFAULT_RUN_DIR,
            mode: str = "evaluate",
            force: bool = False,
            auto_create_subdir: bool = True,
            experiment_name: Optional[str] = None,
            run_name: Optional[str] = None,
    ) -> NoReturn:
        """
        Evaluates task defined by configuration files.
        Builds dataset reader from dataset dataset_config_path and
        substitute generator from substgen_config_path.

        Args:
            substgen_config_path: path to a configuration file.
            dataset_config_path: path to a dataset configuration file.
            run_dir: path to the directory where to store experiment data.
            mode: evaluation mode - 'evaluate' or 'hyperparam_search'
            force: whether to rewrite data in the existing directory.
            auto_create_subdir: if true a subdirectory will be created automatically
                and its name will be the current date and time
            experiment_name: results of the run will be added to 'experiment_name' experiment in MLflow.
            run_name: this run will be marked as 'run_name' in MLflow.
        """
        substgen_config = read_config(substgen_config_path)
        dataset_config = read_config(dataset_config_path)
        config = {
            "class_name": "evaluations.lexsub.LexSubEvaluation",
            "substitute_generator": substgen_config,
            "dataset_reader": dataset_config,
            "verbose": self.verbose,
            "k_list": self.k_list,
            "batch_size": self.batch_size,
            "save_instance_results": self.save_instance_results,
            "save_wordnet_relations": self.save_wordnet_relations,
            "save_target_rank": self.save_target_rank,
        }
        run_dir = Path(run_dir)
        dump_json(Path(run_dir) / "config.json", config)
        if mode == "evaluate":
            # metrics = LexSubEvaluation.evaluate(run_dir=run_dir)
            metrics = self.evaluate_tsar(run_dir)
            print(metrics)
            # runner.evaluate(
            #     config=config,
            #     experiment_name=experiment_name,
            #     run_name=run_name
            # )


if __name__ == "__main__":
    DEFAULT_RUN_DIR = r'../subs/LexSubGen/configs/subst_generators/tsar/'
    sub_generator = SubstituteGenerator.from_config('../subs/LexSubGen/configs/subst_generators/lexsub/xlnet_embs.jsonnet')
    dataset_reader = DatasetReader.from_config('../subs/LexSubGen/configs/subst_generators/dataset/coinco.jsonnet')
    lex_task = LexSubEvaluation(substitute_generator=sub_generator, dataset_reader=dataset_reader)
    lex_task.solve(substgen_config_path='../subs/LexSubGen/configs/subst_generators/lexsub/xlnet_embs.jsonnet', dataset_config_path='../subs/LexSubGen/configs/subst_generators/dataset/coinco.jsonnet',
                   run_dir=DEFAULT_RUN_DIR)