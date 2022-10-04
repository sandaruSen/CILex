import json
import logging
from pathlib import Path
from typing import Dict, List, Any, NoReturn, Optional

from datetime import datetime
import numpy as np
import pandas as pd
from overrides import overrides
from collections import OrderedDict
from tqdm import tqdm
import copy
from sentence_transformers import util

from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.metrics.all_word_ranking_metrics import (
    compute_precision_recall_f1_topk,
    compute_precision_recall_f1_vocab
)
from lexsubgen.metrics.candidate_ranking_metrics import gap_score
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
from lexsubgen.lexsubcon.gloss_score_text import CGLOSS
from lexsubgen.lexsubcon.validation_score import ValidationScore

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

        max_seq_length_gloss = 256
        do_lower_case = False
        max_seq_length = 128
        self.bita = 1
        self.gamma = 0.5
        self.delta = 0.05

        self.fitness_score = False
        self.gloss_score = True
        self.sent_score = False
        self.gloss_score_updated = False

        if self.fitness_score:
            self.validation = ValidationScore(max_seq_length, do_lower_case, pre_trained="bert-large-uncased")
        if self.gloss_score or self.gloss_score_updated:
            self.finding_gloss = CGLOSS('bert-base-uncased', max_seq_length_gloss, do_lower_case)
        if self.sent_score:
            self.similirity_sentence_new = Similarity_new('', '')
            self.similirity_sentence_new.load_model()

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

        definition, score = doc_score_pairs_sorted[0]
        # get which index the definition with highest similarity score is at from all the definitions
        id = docs.index(definition)

        return id, definition, doc_score_pairs

    def get_wordnet_score(self, tokens_lists, target_ids, proposed_words_list, given_words):

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

            gloss_scores.append(score_list)

        return gloss_scores

    def combine_ablation(self, substitues, prob_values, sent_similarity_values, validation_similarities,
                         gloss_similarities):
        final_list = []
        for i in range(len(substitues)):
            item = []
            sub = substitues[i]
            prob = prob_values[i]
            if self.sent_score:
                sim = sent_similarity_values[i]
                for j in range(len(sub)):
                    item.append([sub[j], 0.05 * prob[j] + sim[j]])

            elif self.fitness_score:

                val = validation_similarities[i]
                for j in range(len(sub)):
                    item.append([sub[j], 0.05 * prob[j] + val[j]])

            elif self.gloss_score or self.gloss_score_updated:
                gloss = gloss_similarities[i]
                for j in range(len(sub)):
                    item.append([sub[j], 0.05 * prob[j] + gloss[j]])

            item.sort(key=lambda x: x[1], reverse=True)
            final_list.append([i[0] for i in item])

        return final_list

    @overrides
    def get_metrics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Method for calculating metrics for Lexical Substitution task.

        Args:
            dataset: pandas DataFrame with the whole dataset.
        Returns:
            metrics_data: Dictionary with two keys:

                - all_metrics: pandas DataFrame, extended 'dataset' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
        """

        logger.info(f"Lexical Substitution for {len(dataset)} instances.")

        progress_bar = BatchReader(
            dataset["context"].tolist(),
            dataset["target_position"].tolist(),
            dataset["pos_tag"].tolist(),
            dataset["gold_subst"].tolist(),
            dataset["gold_subst_weights"].tolist(),
            dataset["candidates"].tolist(),
            dataset["target_lemma"].tolist(),
            dataset["given_word"].tolist(),
            batch_size=self.batch_size,
        )

        if self.verbose:
            progress_bar = tqdm(
                progress_bar,
                desc=f"Lexical Substitution for {len(dataset)} instances"
            )

        all_metrics_data, columns = [], None

        for (
                tokens_lists,
                target_ids,
                pos_tags,
                gold_substitutes,
                gold_weights,
                candidates,
                target_lemmas,
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
            text_similarities = []
            validation_similarities = []
            gloss_similarities = []

            cadidate_text_similarities = []
            cadidate_validation_similarities = []
            cadidate_gloss_similarities = []

            if self.sent_score:
                text_similarities = self.get_sentence_Sim_Scores(tokens_lists, target_ids, pred_substitutes,
                                                                 given_words)

            elif self.fitness_score:
                validation_similarities = self.get_validation_score(tokens_lists, target_ids, pred_substitutes,
                                                                    given_words)

            elif self.gloss_score:
                gloss_similarities = self.get_gloss_sim_Scores(tokens_lists, target_ids, pred_substitutes, given_words)

            elif self.gloss_score_updated:
                gloss_similarities = self.get_wordnet_score(tokens_lists, target_ids, pred_substitutes, given_words)

            pred_substitutes = self.combine_ablation(pred_substitutes, prob_values, text_similarities,
                                                     validation_similarities, gloss_similarities)

            if self.sent_score:
                cadidate_text_similarities = self.get_sentence_Sim_Scores(tokens_lists, target_ids, candidates,
                                                                          given_words)

            elif self.fitness_score:
                cadidate_validation_similarities = self.get_validation_score(tokens_lists, target_ids, candidates,
                                                                             given_words)

            elif self.gloss_score:
                cadidate_gloss_similarities = self.get_gloss_sim_Scores(tokens_lists, target_ids, candidates,
                                                                        given_words)

            elif self.gloss_score_updated:
                cadidate_gloss_similarities = self.get_gloss_sim_Scores(tokens_lists, target_ids, candidates,
                                                                        given_words)
            # ------------------------------

            # Ranking candidates using the obtained distribution

            ranked = self.substitute_generator.candidates_for_ablation(
                probs, word2id, candidates, cadidate_text_similarities, cadidate_validation_similarities,
                cadidate_gloss_similarities, self.sent_score, self.fitness_score, self.gloss_score
            )
            ranked_candidates_in_vocab, ranked_candidates = ranked

            for i in range(len(pred_substitutes)):
                instance_results = OrderedDict([
                    ("target_word", tokens_lists[i][target_ids[i]]),
                    ("target_lemma", target_lemmas[i]),
                    ("target_pos_tag", pos_tags[i]),
                    ("target_position", target_ids[i]),
                    ("context", json.dumps(tokens_lists[i])),
                ])

                # Metrics computation
                # Compute GAP, GAP_normalized, GAP_vocab_normalized and ranked candidates
                gap_scores = gap_score(
                    gold_substitutes[i], gold_weights[i],
                    ranked_candidates_in_vocab[i], word2id,
                )
                for metric, gap in zip(self.gap_metrics, gap_scores):
                    instance_results[metric] = gap

                # Computing basic Precision, Recall, F-score metrics
                base_metrics_values = compute_precision_recall_f1_vocab(
                    gold_substitutes[i], word2id
                )
                for metric, value in zip(self.base_metrics, base_metrics_values):
                    instance_results[metric] = value

                # Computing Top K metrics for each K in the k_list
                k_metrics = compute_precision_recall_f1_topk(
                    gold_substitutes[i], pred_substitutes[i], self.k_list
                )
                for metric, value in k_metrics.items():
                    instance_results[metric] = value

                if self.save_instance_results:
                    additional_results = self.create_instance_results(
                        tokens_lists[i], target_ids[i], pos_tags[i],
                        probs[i], word2id, gold_weights[i],
                        gold_substitutes[i], pred_substitutes[i],
                        candidates[i], ranked_candidates[i]
                    )
                    instance_results.update(
                        (k, v) for k, v in additional_results.items()
                    )

                all_metrics_data.append(list(instance_results.values()))

                if columns is None:
                    columns = list(instance_results.keys())

        all_metrics = pd.DataFrame(all_metrics_data, columns=columns)

        mean_metrics = {
            metric: round(all_metrics[metric].mean(skipna=True) * 100, 2)
            for metric in self.metrics
        }

        return {"mean_metrics": mean_metrics, "instance_metrics": all_metrics}

    def create_instance_results(
            self,
            tokens: List[str], target_id: int, pos_tag: str, probs: np.ndarray,
            word2id: Dict[str, int], gold_weights: Dict[str, int],
            gold_substitutes: List[str], pred_substitutes: List[str],
            candidates: List[str], ranked_candidates: List[str],
    ) -> Dict[str, Any]:
        instance_results = OrderedDict()
        pos_tag = to_wordnet_pos.get(pos_tag, None)
        target = tokens[target_id]
        instance_results["gold_substitutes"] = json.dumps(gold_substitutes)
        instance_results["gold_weights"] = json.dumps(gold_weights)
        instance_results["pred_substitutes"] = json.dumps(pred_substitutes)
        instance_results["candidates"] = json.dumps(candidates)
        instance_results["ranked_candidates"] = json.dumps(ranked_candidates)

        if hasattr(self.substitute_generator, "prob_estimator"):
            prob_estimator = self.substitute_generator.prob_estimator
            if target in word2id:
                instance_results["target_subtokens"] = 1
            elif hasattr(prob_estimator, "tokenizer"):
                target_subtokens = prob_estimator.tokenizer.tokenize(target)
                instance_results["target_subtokens"] = len(target_subtokens)
            else:
                instance_results["target_subtokens"] = -1

        if self.save_target_rank:
            target_rank = -1
            if target in word2id:
                target_vocab_idx = word2id[target]
                target_rank = np.where(np.argsort(-probs) == target_vocab_idx)[0][0]
            instance_results["target_rank"] = target_rank

        if self.save_wordnet_relations:
            relations = [
                get_wordnet_relation(target, s, pos_tag)
                for s in pred_substitutes
            ]
            instance_results["relations"] = json.dumps(relations)

        return instance_results

    @overrides
    def dump_metrics(
            self, metrics: Dict[str, Any], run_dir: Path, log: bool = False
    ):
        """
        Method for dumping input 'metrics' to 'run_dir' directory.

        Args:
            metrics: Dictionary with two keys:

                - all_metrics: pandas DataFrame, extended 'dataset' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
            run_dir: Directory path for dumping Lexical Substitution task metrics.
            log: Bool flag for logger.
        """
        if run_dir is not None:
            with (run_dir / "metrics.json").open("w") as fp:
                json.dump(metrics["mean_metrics"], fp, indent=4)
            if self.save_instance_results:
                metrics_df: pd.DataFrame = metrics["instance_metrics"]
                metrics_df.to_csv(run_dir / "results.csv", sep=",", index=False)
                metrics_df.to_html(run_dir / "results.html", index=False)
            if log:
                logger.info(f"Evaluation results were saved to '{run_dir.resolve()}'")
        if log:
            logger.info(json.dumps(metrics["mean_metrics"], indent=4))

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
            metrics = self.evaluate(run_dir)
            print(metrics)
            # runner.evaluate(
            #     config=config,
            #     experiment_name=experiment_name,
            #     run_name=run_name
            # )


if __name__ == "__main__":
    time_format = '%Y_%m_%d_%H_%M'
    start_time = datetime.now()
    # coinco = True
    # sem_eval = False
    #
    # if coinco:
    #     DEFAULT_RUN_DIR = r'/home/users/u7064900/subs/LexSubGen/configs/subst_generators/coinco_con_sent_sim/'
    #     sub_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/lexsub/xlnet_embs.jsonnet'
    #     dataset_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/dataset/coinco.jsonnet'
    #
    #     sub_generator = SubstituteGenerator.from_config(sub_path)
    #     dataset_reader = DatasetReader.from_config(dataset_path)
    #     lex_task = LexSubEvaluation(substitute_generator=sub_generator, dataset_reader=dataset_reader)
    #     lex_task.solve(substgen_config_path=sub_path, dataset_config_path=dataset_path,
    #                    run_dir=DEFAULT_RUN_DIR)
    #
    # elif sem_eval:
    #     DEFAULT_RUN_DIR = r'/home/users/u7064900/subs/LexSubGen/configs/subst_generators/semeval_con_sent_sim'
    #     sub_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/lexsub/xlnet_embs.jsonnet'
    #     dataset_path = '/home/users/u7064900/subs/LexSubGen/configs/subst_generators/dataset/semeval_all.jsonnet'
    #
    #     sub_generator = SubstituteGenerator.from_config(sub_path)
    #     dataset_reader = DatasetReader.from_config(dataset_path)
    #     lex_task = LexSubEvaluation(substitute_generator=sub_generator, dataset_reader=dataset_reader)
    #     lex_task.solve(substgen_config_path=sub_path, dataset_config_path=dataset_path,
    #                    run_dir=DEFAULT_RUN_DIR)

    coinco = True
    sem_eval = False

    if coinco:
        DEFAULT_RUN_DIR = r'C:\Users\admin-u7064900\Desktop\subs_gen\LexSubGen\configs\subst_generators\debug'
        sub_generator = SubstituteGenerator.from_config("xlnet_embs.jsonnet")
        dataset_reader = DatasetReader.from_config("../dataset/coinco.jsonnet")
        lex_task = LexSubEvaluation(substitute_generator=sub_generator, dataset_reader=dataset_reader)
        lex_task.solve(substgen_config_path='xlnet_embs.jsonnet', dataset_config_path='../dataset/coinco.jsonnet',
                       run_dir=DEFAULT_RUN_DIR)

    elif sem_eval:
        DEFAULT_RUN_DIR = r'C:\Users\admin-u7064900\Desktop\subs_gen\LexSubGen\configs\subst_generators\debug'
        sub_generator = SubstituteGenerator.from_config("xlnet_embs.jsonnet")
        dataset_reader = DatasetReader.from_config("../dataset/semeval_all.jsonnet")
        lex_task = LexSubEvaluation(substitute_generator=sub_generator, dataset_reader=dataset_reader)
        lex_task.solve(substgen_config_path='xlnet_embs.jsonnet', dataset_config_path='../dataset/semeval_all.jsonnet',
                       run_dir=DEFAULT_RUN_DIR)

    print('finished the task')
    end_time = datetime.now()
    total_time = end_time - start_time
    print(total_time)

    details = open(DEFAULT_RUN_DIR + "\details.txt", "w")
    details.write("total time: " + str(total_time))
    details.write("\n")
    details.write("sent sim: " + str(lex_task.sent_score))
    details.write("\nvalid sim: " + str(lex_task.fitness_score))
    details.write("\ngloss sim: " + str(lex_task.gloss_score))

# coinco : python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/xlnet_embs.jsonnet --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='coinco_xlnet_embs'
# semeval : python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/xlnet_embs.jsonnet --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_xlnet_embs'
