import logging
import sys
from pathlib import Path
from typing import Dict, NoReturn, Any, Optional

import pandas as pd

from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.params import build_from_config_path

logger = logging.getLogger(Path(__file__).name)


class Task:
    def __init__(
            self,
            substitute_generator: SubstituteGenerator,
            dataset_reader: DatasetReader,
            verbose: bool = False,
    ):
        """
        Base class for performing the evaluation on a particular task.

        Args:
            substitute_generator: Object that generate possible substitutes.
            dataset_reader: Object that can read dataset for Lexical Substitution task.
            verbose: Bool flag for verbosity.
        """
        self.substitute_generator = substitute_generator
        self.dataset_reader = dataset_reader
        self.progress = 0
        self.verbose = verbose

        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        output_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        output_handler.setFormatter(formatter)
        logger.addHandler(output_handler)

    def get_metrics(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Compute particular metrics corresponding to the Task.

        Args:
            dataset: pandas DataFrame with whole dataset.
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config_path: str):
        """
        Builds an object of a class that inherits from this class
            using parameters described in a config file
        Args:
            config_path: path to .jsonnet config.
                For example, see this config for LexSubEvaluation class:
                "configs/evaluations/lexsub/semeval_all_elmo.jsonnet"
        Returns: an object that was created with the parameters described in the given config
        """
        evaluation_object, _ = build_from_config_path(config_path)
        return evaluation_object

    def dump_metrics(
            self, metrics: Dict[str, Any], run_dir: Optional[Path] = None, log: bool = False
    ) -> NoReturn:
        """
        Method for dumping input 'metrics' to 'run_dir' directory.

        Args:
            metrics: Dictionary that maps metrics name to their values.
            run_dir: Directory path for dumping Lexical Substitution task metrics.
            log: Bool flag for logger.
        """
        raise NotImplementedError

    def evaluate(self, run_dir: Optional[Path] = None) -> NoReturn:
        """
        Method for running Task evaluation.

        Args:
            run_dir: Directory path for dumping results of the evaluation.
        """
        dataset = self.dataset_reader.read_dataset()[:2]
        metrics = self.get_metrics(dataset)
        self.dump_metrics(metrics, run_dir, log=self.verbose)
        return metrics


    def get_wordnet_pos(self,treebank_tag):
        from nltk.corpus import wordnet
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


    def get_pos(self, row):
        import nltk
        sentence = row['sentence']
        word = row['target_word']
        index = row['target_position']
        pos_tags = [p for w, p in nltk.pos_tag(nltk.word_tokenize(sentence)) if w == word][0]
        pos = self.get_wordnet_pos(pos_tags)
        return pos

    def get_id(self,row):
        sentence = row['sentence']
        context = row['context']
        word = row['target_word']
        index = context.index(word)
        return index

    def clear(self,row):
        value = row['sentence']
        value = value.replace("\\", "")
        return value

    def given_word(self,row):
        target_word = row['target_word']
        pos = row['pos_tag']
        return target_word+'.'+pos

    def evaluate_tsar(self,run_dir) -> NoReturn:
        """
        Method for running Task evaluation.

        Args:
            run_dir: Directory path for dumping results of the evaluation.
        """

        file_path = r'C:\Users\admin-u7064900\Desktop\subs_gen\LexSubGen\tsar\tsar2022_en_trial_none.tsv'
        colnames = ['sentence', 'target_word']
        df = pd.read_csv(file_path, sep="\t", names=colnames)

        df['sentence'] = df.apply(self.clear, axis=1)
        df['context'] = df['sentence'].apply(lambda x: x.split())
        df['target_position'] = df.apply(self.get_id, axis=1)
        df['pos_tag'] = df.apply(self.get_pos, axis=1)
        df['given_word'] = df.apply(self.given_word, axis=1)


        dataset = df
        metrics = self.get_metrics(dataset)
        return dataset


# evaluate_tsar('')
