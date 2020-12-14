import re
import logging
from copy import deepcopy

import numpy as np

from typing import List

from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.message_processing import Message, extract_tokens, parse_text, read_data


def drop_path_in_code(parsed: list):
    result = []
    for item in parsed:
        if item['type'] in ['code', 'quoted']:
            code = item['text']
            code = re.sub('Users/.*/', '', code)
            code = re.sub('Users\\\\.*\\\\', '', code)
            code = re.sub('user/.*/', '', code)
            item['text'] = code
            result.append(item)
        else:
            result.append(item)
    return result


def extract_code(messages: List[Message]):
    code_samples = []
    for message in messages:
        code_samples.append(extract_tokens(drop_path_in_code(parse_text(message.text)), targets=('quoted', 'code')))
    return code_samples


class ErrorCodeSimilarityModel:

    def __init__(self, max_df=1.0, ngram_range=(1, 3)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df)
        self.messages_list = None
        self.vector_matrix = None

    def train(self, messages_list: List[Message]):
        self.messages_list = deepcopy(messages_list)
        code = list(map(lambda x: ' '.join(x), extract_code(messages_list)))
        try:
            self.vector_matrix = self.vectorizer.fit_transform(code)
        except ValueError:
            logging.info("WARNING NO CODE IN MESSAGES, so code model is not working.")
            self.vector_matrix = None

    def compare_messages(self, first: Message, second: Message):
        return cosine_similarity([self.get_vector(first)], [self.get_vector(second)])[0, 0]

    def get_vector(self, message: Message):
        code = ' '.join(extract_code([message])[0])
        if self.vector_matrix is not None:
            return self.vectorizer.transform([code]).toarray()[0]
        return [0]

    def find_similars(self, message: Message, count=5):
        if self.vector_matrix is not None:
            vector = self.get_vector(message)
            similarity = cosine_similarity([vector], self.vector_matrix)[0]
            top_count_idx = np.argsort(similarity)[-count:]
            return [(similarity[index], self.messages_list[index]) for index in reversed(top_count_idx)]
        return []

    def update_model(self, message):
        vector = self.get_vector(message)
        if self.vector_matrix is None and vector != [0]:
            self.vector_matrix = csr_matrix([vector])
            self.messages_list.append(message)
        elif self.vector_matrix is not None:
            self.vector_matrix = vstack([self.vector_matrix, csr_matrix([vector])])
            self.messages_list.append(message)


def test_text_model():
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    csm = ErrorCodeSimilarityModel()
    csm.train(messages)
    new_msg = Message('```Test one two```', '', '', '')
    csm.update_model(new_msg)
    print(csm.find_similars(new_msg)[0][1])


if __name__ == '__main__':
    test_text_model()
