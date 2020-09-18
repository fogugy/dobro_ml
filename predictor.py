import pickle
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer


class Predictor:
    model_path = 'xgb.model'
    columns_path = 'columns.pickle'
    hash_vec_path = 'hash_vec.pickle'

    re_punct = re.compile(f'[{string.punctuation}– ]+')
    russian_stopwords = stopwords.words("russian")
    morpher = MorphAnalyzer()

    _model = None
    _columns = []
    _hash_vec = None

    def __init__(self):
        self.init()

    def init(self):
        with open(self.model_path, 'rb') as f:
            self._model = pickle.load(f)

        with open(self.columns_path, 'rb') as f:
            self._columns = pickle.load(f)

        with open(self.hash_vec_path, 'rb') as f:
            self._hash_vec = pickle.load(f)

    def predict(self, text):
        text = self._prepare_text(text)
        df = pd.DataFrame({'text': [text]})

        # part of speech features
        df[self._columns] = np.zeros((1, len(self._columns)))
        pos_dict = self._pos(text)
        for key, value in pos_dict.items():
            if key in self._columns:
                df.iloc[0][key] = value

        # hashing features
        h = np.array(self._hash_vec.transform([text]).toarray())

        for x in range(h.shape[1]):
            df[f'hvec_{x}'] = h[:, x]

        return self._model.predict(df.drop(['text'], axis=1))[0] > 0.5

    def _prepare_text(self, text):
        text = self._punct(text)
        t_arr = self._clear(text.split())
        return ' '.join(t_arr)

    def _hash(self, text):
        return

    def _punct(self, text):
        return re.sub(self.re_punct, ' ', text)

    def _clear(self, arr):
        return [w for w in arr if not w in self.russian_stopwords]

    def _pos(self, text):
        tags = [str(self.morpher.parse(w)[0].tag) for w in text.split(' ')]
        return Counter(tags)
