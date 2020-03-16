import os

import numpy
import torch

from new_pretrained.configuration_roberta import RobertaConfig
from new_pretrained.configuration_xlm import XLMConfig
from new_pretrained.configuration_xlnet import XLNetConfig
from new_pretrained.modeling_roberta import RobertaModel
from new_pretrained.modeling_xlm import XLMModel
from new_pretrained.modeling_xlnet import XLNetModel
from new_pretrained.tokenization_roberta import RobertaTokenizer
from new_pretrained.tokenization_xlm import XLMTokenizer
from new_pretrained.tokenization_xlnet import XLNetTokenizer
from preprocess import Vectorizer, index_sentence


class RoBERTa(Vectorizer):
    def __init__(self):
        self.config = RobertaConfig.from_pretrained("roberta-base")
        self.config.output_hidden_states = True

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
        self.model = RobertaModel.from_pretrained("roberta-base", config=self.config)
        self.model.eval()

    def vectorize(self, sentence):
        """
        Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
        TODO: Figure out if this is correct.
        """
        tokens_tensor = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)

        with torch.no_grad():
            sequence_output, pooled_output, hidden_states = self.model(tokens_tensor)

        # TODO: What about sequence output and pooled ouptut? hidden states is 13 layers,
        # but bert had input embeddings + embeddings?
        embeddings = torch.stack(hidden_states, dim=0).squeeze()[:, :, :]
        embeddings = embeddings.detach().numpy()

        return embeddings


class XLM(Vectorizer):
    def __init__(self):
        self.config = XLMConfig.from_pretrained("xlm-mlm-enfr-1024")
        self.config.output_hidden_states = True

        self.tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-enfr-1024", do_lower_case=False)
        self.model = XLMModel.from_pretrained("xlm-mlm-enfr-1024", config=self.config)
        self.model.eval()

    def vectorize(self, sentence):
        """
        Return a tensor representation of the sentence of size (7 layers, num tokens, 1024 dim).
        TODO: Figure out if this is correct.
        """
        tokens_tensor = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)

        with torch.no_grad():
            pred_scores, hidden_states = self.model(tokens_tensor)

        embeddings = torch.stack(hidden_states, dim=0).squeeze()[:, :, :]
        embeddings = embeddings.detach().numpy()

        return embeddings


class XLNet(Vectorizer):
    def __init__(self):
        self.config = XLNetConfig.from_pretrained("xlnet-base-cased")
        self.config.output_hidden_states = True

        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=False)
        self.model = XLNetModel.from_pretrained("xlnet-base-cased", config=self.config)
        self.model.eval()

    def vectorize(self, sentence):
        """
        Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
        TODO: Figure out if this is correct.
        """
        tokens_tensor = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)

        with torch.no_grad():
            last_hidden_state, hidden_states = self.model(tokens_tensor)

        embeddings = torch.stack(hidden_states, dim=0).squeeze()[:, :, :]
        embeddings = embeddings.detach().numpy()

        return embeddings


if __name__ == '__main__':
    EMBEDDINGS_PATH = "/tmp/f_contextual_embeddings/"

    roberta = RoBERTa()
    sentences = index_sentence('sts.csv', 'roberta/word2sent.json', roberta.tokenizer.tokenize)
    roberta.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'roberta.hdf5'))

    #xlm = XLM()
    #sentences = index_sentence('sts.csv', 'xlm/word2sent.json', xlm.tokenizer.tokenize)
    #xlm.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'xlm.hdf5'))

    #xlnet = XLNet()
    #sentences = index_sentence('sts.csv', 'xlnet/word2sent.json', xlnet.tokenizer.tokenize)
    #xlnet.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'xlnet.hdf5'))
