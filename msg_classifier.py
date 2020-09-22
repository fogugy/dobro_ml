import pickle
import re

import numpy as np
import pandas as pd
from html.parser import HTMLParser
from nltk.corpus import stopwords
from pymystem3 import Mystem

import gensim.downloader as api

html_parser = HTMLParser()

m = Mystem()
stop_rus = stopwords.words('russian')

w2v = api.load("word2vec-ruscorpora-300")

with open('./d_associated.pickle', 'rb') as f:
    d_associated = pickle.load(f)

emoticon_dict = {
    ")": "радость",
    ":)": "радость",
    ":‑)": "радость",
    ":-]": "радость",
    ":-3": "радость",
    ":->": "радость",
    "8-)": "радость",
    ":-}": "радость",
    ":o)": "радость",
    ":c)": "радость",
    ":^)": "радость",
    "=]": "радость",
    "=)": "радость",
    "<3": "радость",
    "(": "печаль",
    ":-(": "печаль",
    ":(": "печаль",
    ":c": "печаль",
    ":<": "печаль",
    ":[": "печаль",
    ">:[": "печаль",
    ":{": "печаль",
    ">:(": "печаль",
    ":-c": "печаль",
    ":-< ": "печаль",
    ":-[": "печаль",
    ":-||": "печаль"
}

pos_conv = {
    "A": "ADJ",
    "ADV": "ADV",
    "ADVPRO": "ADV",
    "ANUM": "ADJ",
    "APRO": "DET",
    "COM": "ADJ",
    "CONJ": "SCONJ",
    "INTJ": "INTJ",
    "NONLEX": "X",
    "NUM": "NUM",
    "PART": "PART",
    "PR": "ADP",
    "S": "NOUN",
    "SPRO": "PRON",
    "UNKN": "X",
    "V": "VERB",
}

default_word = 'человек_NOUN'


def prepare(text):
    text = html_parser.unescape(text)
    text = text.lower()
    text = translateDict(emoticon_dict, text)
    text = re.sub("@[\w]*", "пользователь", text)
    text = re.sub("[A-Za-z]+", " ", text)
    text = clearPunct(text)
    text = ' '.join([w for w in text.split() if (len(w) > 1 and w not in stop_rus)])
    return text


def translateDict(dictionary, text):
    return " ".join([dictionary[x] if x in dictionary else x for x in re.split(r"[\s\.,;:]+", text)])


def clearPunct(text):
    return re.sub(r"[\W\s0-9_]+", " ", text)


def tag(word):
    try:
        processed = m.analyze(word)[0]
        lemma = processed["analysis"][0]["lex"].lower().strip()
        pos = processed["analysis"][0]["gr"].split(',')[0]
        pos = pos.split('=')[0].strip()
        tagged = f"{lemma}_{pos_conv.get(pos) or 'NOUN'}"
        return tagged
    except:
        return default_word


def get_vec(text, agg_func):
    v = []
    for w in text.split(' '):
        tagged = tag(w)
        v.append(w2v.get_vector(tagged) if w2v.vocab.get(tagged) else w2v.get_vector('человек_NOUN'))
    return agg_func(np.array(v))


def _avg(v):
    return np.sum(v, axis=0) / v.shape[0]


def _max(v):
    return np.max(v, axis=0)


def _min(v):
    return np.min(v, axis=0)


def get_feat(text, aggr):
    if type(text) is not str:
        text = default_word
    vec = get_vec(text, _min).tolist() + get_vec(text, _max).tolist() + get_vec(text, _avg).tolist()
    return vec


def has_abscent(text):
    _min = 1
    for word in text:
        tagged = tag(word)
        if tagged in d_associated:
            return 1
        if tagged in w2v.vocab:
            _min = min(_min, w2v.distances(tagged, d_associated).min())
    return int(_min <= 0.5)


example_v = w2v.get_vector(default_word)


class MsgClassifier:
    model_path = 'msg_classifier.pickle'
    columns_path = 'msg_columns.pickle'
    hash_vec_path = 'msg_hash_vec.pickle'

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
        text = prepare(text)

        if has_abscent(text.split(' ')):
            return [0, 1]

        df = pd.DataFrame(columns=self._columns)

        # part of speech features
        df[self._columns] = np.zeros((1, len(self._columns)))

        v = get_vec(text, _min).tolist() + get_vec(text, _max).tolist() + get_vec(text, _avg).tolist()

        # hashing features
        h = self._hash_vec.transform([text]).toarray().flatten().tolist()

        df.loc[0] = v + h

        return self._model.predict_proba(df)[0]
