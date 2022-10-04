import json
import logging
from pathlib import Path
from typing import Dict, List, Any, NoReturn, Optional

import numpy as np
import pandas as pd
from fire import Fire
from overrides import overrides
from collections import OrderedDict
from tqdm import tqdm
import copy

from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.metrics.all_word_ranking_metrics import (
    compute_precision_recall_f1_topk,
    compute_precision_recall_f1_vocab
)
from lexsubgen.metrics.candidate_ranking_metrics import gap_score
from lexsubgen.runner import Runner
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from lexsubgen.utils.file import dump_json
from lexsubgen.utils.params import read_config
from lexsubgen.utils.wordnet_relation import to_wordnet_pos, get_wordnet_relation

# lexsub con imports

from lexsubgen.lexsubcon.noise import Gloss_noise
from lexsubgen.lexsubcon.wordnet import Wordnet
from lexsubgen.lexsubcon.score import Cmasked
from lexsubgen.lexsubcon.similarity_new_predict import Similarity_new

logger = logging.getLogger(Path(__file__).name)

DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent.parent / "debug" / Path(__file__).stem


class LexSubEvaluation(Task):
    def __init__(
            self,
            substitute_generator: SubstituteGenerator = None,
            dataset_reader: DatasetReader = None,
            verbose: bool = True,
            k_list: List[int] = (1, 3, 10),
            batch_size: int = 50,
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
        self.train_file = ''
        self.train_golden_file = ''
        self.similarity_path_save = '../../../lexsubgen/lexsubcon/checkpoint/similarity_new_bert/'
        self.similirity_sentence_new = Similarity_new(self.train_file, self.train_golden_file)
        # self.similirity_sentence_new.load_model(model=self.similarity_path_save)
        self.similirity_sentence_new.load_model()
        self.bita = 1

    def get_proposal_score_sent_similarity_score(self, tokens_lists, target_ids, given_words):
        for i in range(len(tokens_lists)):

            text = tokens_lists[i]
            original_text = text
            index_word = target_ids[i]
            change_word = tokens_lists[i][int(index_word)]
            synonyms = []

            main_word = given_words[i]

            if main_word.split('.')[0] == "":
                word_temp = "."
            else:
                word_temp = main_word.split('.')[0]

            proposed_words = self.noise_gloss.created_proposed_list(word_temp, self.wordnet_gloss,
                                                                    main_word.split('.')[-1])

            if len(proposed_words) > 30:
                pass

            else:
                if self.noise_type == "GLOSS":
                    """
                    find the probable gloss of each word
                    """
                    if len(synonyms) == 0:
                        if main_word.split('.')[0] == "":
                            word_temp = "."
                        else:
                            word_temp = main_word.split('.')[0]
                        synonyms = self.noise_gloss.adding_noise(word_temp,
                                                                 self.wordnet_gloss,
                                                                 main_word.split('.')[-1])
                        try:
                            synonyms.remove(main_word.split('.')[0])
                        except:
                            pass

                    if len(synonyms) == 0:
                        # 91- do not have wordnet synonyms in LS14
                        noise_type = "GAUSSIAN"

                proposed_words = self.proposal.proposed_candidates(original_text, change_word, int(index_word),
                                                                   noise_type=self.noise_type, synonyms=synonyms,
                                                                   proposed_words_temp=proposed_words, top_k=30)

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

    def combine_xlnet_sent_scores(self, substitues, prob_values, sent_similarity_values):
        final_list = []
        for i in range(len(substitues)):
            item = []
            sub = substitues[i]
            prob = prob_values[i]
            sim = sent_similarity_values[i]
            for j in range(len(sub)):
                item.append([sub[j], prob[j] + sim[j]])

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
            # compute sentence similarity for the selected terms
            # self.get_proposal_score_sent_similarity_score(tokens_lists, target_ids, given_words)


            # text_similarities = self.get_sentence_Sim_Scores(tokens_lists, target_ids, pred_substitutes, given_words)
            # pred_substitutes = self.combine_xlnet_sent_scores(pred_substitutes, prob_values, text_similarities)
            pred_substitutes = [i[:10] for i in pred_substitutes]


            for j in range(10):
                temp = []
                for i in pred_substitutes:
                    temp.append(i[j])
                dataset[str(j)] = temp

            dataset.drop(['context', 'target_position','pos_tag','given_word'], axis=1, inplace=True)

            print(dataset)

            dataset.to_csv('../../../tsar/test.tsv', sep='\t', index=False, header=False)

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

    DEFAULT_RUN_DIR = r'C:\Users\admin-u7064900\Desktop\subs_gen\LexSubGen\configs\subst_generators\debug'
    sub_generator = SubstituteGenerator.from_config("xlnet_embs.jsonnet")
    dataset_reader = DatasetReader.from_config("../dataset/coinco.jsonnet")
    lex_task = LexSubEvaluation(substitute_generator=sub_generator, dataset_reader=dataset_reader)
    lex_task.solve(substgen_config_path='xlnet_embs.jsonnet', dataset_config_path='../dataset/coinco.jsonnet',
                       run_dir=DEFAULT_RUN_DIR)



