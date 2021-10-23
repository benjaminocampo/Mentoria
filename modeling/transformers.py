import fasttext
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode
from nltk import (corpus, tokenize, download)
from encoding import encode_labels

download("stopwords")
download('punkt')

class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class FastTextVectorizer(BaseTransformer):
    def __init__(self,
                 corpus_file="titles.txt",
                 pretext_task="cbow",
                 lr=0.3,
                 epoch=100,
                 dim=100,
                 wordNgrams=3,
                 ws=3):
        self.corpus_file = corpus_file
        self.pretext_task = pretext_task
        self.lr = lr
        self.epoch = epoch
        self.dim = dim
        self.wordNgrams = wordNgrams
        self.ws = ws
        self.model = None        

    def get_sequence_vector(self, sentence):
        return self.model.get_sentence_vector(sentence)

    def unsupervised_data_gen(self, sentences):
        with open(self.corpus_file, "w") as out:
            for s in sentences:
                out.write(s + "\n")

    def fit(self, sentences, y=None):
        self.unsupervised_data_gen(sentences)
        self.model = fasttext.train_unsupervised(self.corpus_file,
                                                 model=self.pretext_task,
                                                 lr=self.lr,
                                                 epoch=self.epoch,
                                                 dim=self.dim,
                                                 wordNgrams=self.wordNgrams,
                                                 ws=self.ws)
        return self

    def transform(self, sentences, y=None):
        return np.vstack(sentences.apply(self.get_sequence_vector))


class Preprocesser(BaseTransformer):
    @staticmethod
    def remove_contractions(s):
        # TODO: See if we can find more contractions.
        contractions = [
            ("c/u", "cada uno"),
            ("c/", "con"),
            ("p/", "para"),
        ]
        for expression, replacement in contractions:
            s = s.replace(expression, replacement)
        return s

    @staticmethod
    def remove_numbers(s):
        numbers = r"\d+"
        return re.sub(numbers, "", s)

    @staticmethod
    def remove_symbols(s):
        symbols = r"[^a-zA-Z ]"
        return re.sub(symbols, "", s)

    @staticmethod
    def remove_stopwords(s, language):
        s = ' '.join(w for w in tokenize.word_tokenize(s)
                     if not w in corpus.stopwords.words(language))
        return s

    def clean_text(self, s: str, language: str) -> str:
        """
        Given a string @s and its @language the following parsing is performed:
            - Lowercase letters.
            - Removes non-ascii characters.
            - Removes numbers and special symbols.
            - Expand common contractions.
            - Removes stopwords.
        """
        s = unidecode(s)
        s = s.lower()
        s = self.remove_contractions(s)
        s = self.remove_numbers(s)
        s = self.remove_symbols(s)
        s = self.remove_stopwords(s, language)
        return s

    
    def transform(self, df):
        # Remove numbers, symbols, special chars, contractions, etc.
        cleaned_title_col = df[["title", "language"
                                          ]].apply(lambda x: self.clean_text(*x),
                                                   axis=1)
        # Add @cleaned_title column to df
        df = df.assign(cleaned_title=cleaned_title_col)
        df = df.assign(
            encoded_category=encode_labels(df["category"]))
        df = df.dropna()
        return df