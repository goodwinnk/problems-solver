from copy import deepcopy
from typing import List

import logging
import numpy as np
from nltk import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, vstack
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
        self.messages_list = deepcopy(messages_list)
        normalized = self.normalize_data(messages_list)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(map(lambda x: ' '.join(x), normalized))
            logging.info(f"Text model are training on {len(messages_list)} samples")
            if tfidf_matrix.shape[1] <= self.svd.n_components:
                self.svd.set_params(n_components=tfidf_matrix.shape[1] - 2)
                logging.warning(f'Too little message for the SVD model, now n_components={tfidf_matrix.shape[1] - 2}')
            self.vector_matrix = self.svd.fit_transform(tfidf_matrix)
        except ValueError:
            logging.info("WARNING NO MESSAGES, so text model is not working.")
            self.vector_matrix = None

    def get_vector(self, message: Message):
        if self.vector_matrix is not None:
            normalized = ' '.join(self.normalize_data([message])[0])
            vector = self.vectorizer.transform([normalized])
            return self.svd.transform(vector)[0]
        logging.warning("No messages in text model, all vectors now are [0]")
        return [0]

    def compare_messages(self, first: Message, second: Message):
        return cosine_similarity([self.get_vector(first)], [self.get_vector(second)])[0, 0]

    def find_similars(self, message: Message, count=5):
        vector = self.get_vector(message)
        if self.vector_matrix is not None:
            similarity = cosine_similarity([vector], self.vector_matrix)[0]
            count = count if similarity.shape[0] >= count else similarity.shape[0]
            top_count_idx = np.argsort(similarity)[-count:]
            return [(similarity[index], self.messages_list[index]) for index in reversed(top_count_idx)]
        return []

    def update_model(self, message: Message):
        vector = self.get_vector(message)
        self.vector_matrix = vstack([self.vector_matrix, csr_matrix([vector])])
        self.messages_list.append(message)


def test_text_model():
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    tsm = TextSimilarityModel(max_df=1.0)
    tsm.train(messages)
    new_msg = Message('Test one two three hello, world', 'test', 'test', '')
    tsm.update_model(new_msg)
    print(tsm.find_similars(new_msg)[0][1])

if __name__ == '__main__':
    test_text_model()
