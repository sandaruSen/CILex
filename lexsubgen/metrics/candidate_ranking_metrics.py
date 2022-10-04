import re
from typing import List, Tuple, Dict, Union, Set, Any

import numpy as np

WORD_PAT_RE = re.compile(r"^[A-Za-z]+$")


def gap_score(
        gold_substitutes: List[str],
        gold_weights: List[int],
        ranked_candidates: List[str],
        vocabulary: Union[Set[str], Dict[str, Any]],
) -> Tuple[float, float, float]:
    """
    Function for calculating GAP metric.
    Example:
        gold_substitutes = ["intelligent", "clever"]
        gold_weights = [3, 2]
        ranked_candidates = ["positive", "smart", "clever", "intelligent", "talented"]
        vocabulary = {
            "happy": 0, "bright": 1, "positive": 2, "intelligent": 3,
            "clever": 4, "smart": 5, "talented": 6, "curious": 7,
            ....
            "watch": 31998, "dogs": 31999
        }

        GAP is described here in Section 4.2 in https://www.aclweb.org/anthology/P10-1097.pdf.

    Args:
        gold_substitutes: substitutes generated by annotators
        gold_weights: corresponding number of annotators for each substitute
        ranked_candidates: list of ranked candidates
        vocabulary: python set or dictionary (mapping from words to their indices in the vocabulary)
    Returns:
        function returns 3 metrics (3 float values):
            GAP - base metric for candidate-ranking task.
            GAP_normalized - similar to GAP metric, but we exclude MWE (multi word expressions) from gold_substitutes
            GAP_vocab_normalized - similar to GAP metric, but we exclude all OOV words from gold_substitutes
    """

    # GAP with MWEs
    gold_map = {word: weight for word, weight in zip(gold_substitutes, gold_weights)}
    gap = compute_gap(gold_map, ranked_candidates)

    # GAP without MWEs
    gold_map_normalized = {
        word: weight
        for word, weight in zip(gold_substitutes, gold_weights)
        if WORD_PAT_RE.match(word)
    }
    normalized_candidates = [w for w in ranked_candidates if WORD_PAT_RE.match(w)]
    gap_normalized = compute_gap(
        gold_map_normalized,
        normalized_candidates,
    )

    # GAP without OOV words
    gold_map_vocab_normalized = {
        word: weight
        for word, weight in zip(gold_substitutes, gold_weights)
        if word in vocabulary
    }
    vocab_normalized_candidates = [
        w for w in ranked_candidates if w in vocabulary
    ]
    gap_vocab_normalized = compute_gap(
        gold_map_vocab_normalized,
        vocab_normalized_candidates,
    )
    return gap, gap_normalized, gap_vocab_normalized


def compute_gap_nominator(
        ranked_candidates: List[str],
        gold2weight: Dict[str, int],
) -> float:
    """
    Method for computing nominator for GAP score.

    Args:
        ranked_candidates: List of ranked candidates.
        gold2weight: Dictionary that maps gold word to its annotators number.
    Returns:
        nominator: computed nominator of GAP score.
    """
    cumsum = 0.0
    nominator = 0.0
    for rank, word in enumerate(ranked_candidates):
        weight = gold2weight.get(word, 0)
        if weight:
            cumsum += weight
            nominator += cumsum / (rank + 1)
    return nominator


def compute_gap(
        gold_mapping: Dict[str, int],
        ranked_candidates: List[str],
) -> Union[float, None]:
    """
    Method for computing GAP metric.

    Args:
        gold_mapping: Dictionary that maps gold word to its annotators number.
        ranked_candidates: List of ranked candidates.
    Returns:
        gap: computed GAP score.
    """
    if not gold_mapping:
        return None

    cumsum = np.cumsum(list(gold_mapping.values()))
    arange = np.arange(1, len(gold_mapping) + 1)
    gap_denominator = (cumsum / arange).sum()

    gap_nominator = compute_gap_nominator(ranked_candidates, gold_mapping)

    return gap_nominator / gap_denominator



#
# gold_substitutes = ["intelligent", "clever"]
# gold_weights = [3, 2]
# ranked_candidates = ["positive", "smart", "clever", "intelligent", "talented"]
# vocabulary = {
#     "happy": 0, "bright": 1, "positive": 2, "intelligent": 3,
#     "clever": 4, "smart": 5, "talented": 6, "curious": 7,
#     "watch": 31998, "dogs": 31999
# }
#
# gap, gap_normalized, gap_vocab_normalized = gap_score(gold_substitutes, gold_weights, ranked_candidates, vocabulary)
# print(gap_normalized)
#
# from lexsubgen.lexsubcon.generalized_average_precision import GeneralizedAveragePrecision
#
# def gap_calculation( gold_candidates, output_candidates):
#     ignored = 0
#     i = 0
#     sum_gap = 0.0
#
#     randomize = False
#     # how to go over the evaluation results
#     for j in range(len(gold_candidates)):
#         gold_weights = gold_candidates[j]
#         eval_weights = output_candidates[j]
#         gap = GeneralizedAveragePrecision.calc(gold_weights, eval_weights, randomize)
#         if (gap < 0):
#             # this happens when there is nothing left to rank after filtering the multi-word expressions
#             ignored += 1
#             continue
#         # out_file.write(str(j) + "\t" + str(gap) + "\n")
#         i += 1
#         sum_gap += gap
#
#     mean_gap = sum_gap / i
#
#     print(mean_gap)
#
#
#
#
# gold_value_list = []
# pred_value_list = []
# for j in range(len(gold_substitutes)):
#     gold_value_list.append((gold_substitutes[j], gold_weights[j]))
#
# length = len(ranked_candidates)
# for j in range(len(ranked_candidates)):
#     pred_value_list.append((ranked_candidates[j], length - j))
#
# gap_calculation([gold_value_list], [pred_value_list])