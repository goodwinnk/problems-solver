from typing import List

import nltk
import numpy as np
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.corpus import DummyStemmer
from nlp.message_processing import Message, extract_tokens, parse_text, read_data


def normalize(text, stopwords=(), stemmer=None, use_quoted=True):
    stemmer = stemmer if stemmer else DummyStemmer()
    stemmed_stopwords = set(stemmer.stem(word) for word in stopwords)
    result = []
    for token in extract_tokens(parse_text(text), targets=['text', 'link', '????' if not use_quoted else 'quoted']):
        stemmed = stemmer.stem(token)
        if stemmed not in stemmed_stopwords:
            result.append(stemmed)
    return result


class TextSimilarityModel:

    def __init__(self, max_df=1.0, n_components=100, ngram_range=(1, 3)):
        self.stemmer = PorterStemmer()
        # nltk.download()  # TODO make it one time
        self.stopwords = stopwords.words('english')
        self.svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df)
        self.messages_list = None
        self.vector_matrix = None

    def normalize_data(self, messages_list):
        return list(map(lambda m: normalize(m.text, self.stopwords, self.stemmer, True), messages_list))

    def len_comparision(self, first, second):
        lens = list(map(len, self.normalize_data([first, second])))
        return min(lens[0] / lens[1], lens[1] / lens[0]) if lens[0] and lens[1] else 0

    def train(self, messages_list: List[Message]):
        self.messages_list = messages_list
        normalized = self.normalize_data(messages_list)
        tfidf_matrix = self.vectorizer.fit_transform(map(lambda x: ' '.join(x), normalized))
        self.vector_matrix = self.svd.fit_transform(tfidf_matrix)

    def get_vector(self, message: Message):
        normalized = ' '.join(self.normalize_data([message])[0])
        vector = self.vectorizer.transform([normalized])
        return self.svd.transform(vector)[0]

    def compare_messages(self, first: Message, second: Message):
        return cosine_similarity([self.get_vector(first)], [self.get_vector(second)])[0, 0]

    def find_similars(self, message: Message, count=5):
        vector = self.get_vector(message)
        similarity = cosine_similarity([vector], self.vector_matrix)[0]
        top_count_idx = np.argsort(similarity)[-count:]
        return [(similarity[index], self.messages_list[index]) for index in reversed(top_count_idx)]


def test_text_model():
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    tsm = TextSimilarityModel(max_df=1.0)
    tsm.train(messages)


if __name__ == '__main__':
    test_text_model()
