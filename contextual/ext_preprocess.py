from new_pretrained.modeling_roberta import RobertaModel
from new_pretrained.tokenization_roberta import RobertaTokenizer
from new_pretrained.modeling_xlm import XLMModel
from new_pretrained.tokenization_xlm import XLMTokenizer

from preprocess import Vectorizer, index_sentence

import os
import numpy
import torch


class RoBERTa(Vectorizer):
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.eval()

    def vectorize(self, sentence):
        # TODO: return embeddings of RoBERTa
        pass


class XLM(Vectorizer):
    def __init__(self):
        self.tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024')
        self.model = XLMModel.from_pretrained('xlm-mlm-enfr-1024')
        self.model.eval()

    def vectorize(self, sentence):
        # TODO: return embeddings of XLM
        pass


if __name__ == '__main__':
    # where to save the contextualized embeddings
    EMBEDDINGS_PATH = "./contextual_embeddings"

    # sts.csv has been preprocessed to remove all quotes of type ", since they are often not completed
    roberta = RoBERTa()
    sentences = index_sentence('sts.csv', 'roberta/word2sent.json', roberta.tokenizer.tokenize)
    roberta.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'roberta.hdf5'))

    xlm = XLM()
    sentences = index_sentence('sts.csv', 'xlm/word2sent.json', xlm.tokenizer.tokenize)
    xlm.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'xlm.hdf5'))
