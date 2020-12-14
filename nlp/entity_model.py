from copy import deepcopy

import numpy as np

import nltk
from nltk.corpus import words

from nlp.message_processing import Message, extract_tokens, parse_text, read_data

from typing import List, Set


class EntitySimilarityModel:

    def __init__(self):
        self.messages_list = None
        self.entity_matrix = None
        try:
            'hi' in words.words()
        except LookupError:
            nltk.download('corpus')
        self.stemmer = nltk.PorterStemmer()
        self.vocabulary = None

    def extract_entities(self, messages_list) -> List[Set]:
        result = []
        for msg in messages_list:
            text = msg.text
            result.append([])
            tokens = extract_tokens(parse_text(text), targets=['text', 'link', 'quoted'])
            for token in tokens:
                token = token.lower()
                if self.stemmer.stem(token) not in self.vocabulary:
                    result[-1].append(token)
            result[-1] = set(result[-1])
        return result

    def train(self, messages_list: List[Message]):
        self.messages_list = deepcopy(messages_list)
        self.vocabulary = set(self.stemmer.stem(word) for word in words.words())
        self.entity_matrix = self.extract_entities(messages_list)

    def get_vector(self, message: Message):
        return set(self.extract_entities([message])[0])

    def jaccar_index(self, first: set, second: set):
        if len(first) + len(second):
            return len(first.intersection(second)) / len(first.union(second))
        return 0

    def compare_messages(self, first: Message, second: Message):
        return self.jaccar_index(self.get_vector(first), self.get_vector(second))

    def find_similars(self, message: Message, count=5):
        vector = self.get_vector(message)

        def dist_func(entities: set):
            return self.jaccar_index(vector, entities)

        similarity = np.array(list(map(dist_func, self.entity_matrix)), dtype=object)
        top_count_idx = np.argsort(similarity)[-count:]
        return [(similarity[index], self.messages_list[index]) for index in reversed(top_count_idx)]

    def update_model(self, message):
        vector = self.get_vector(message)
        self.entity_matrix.append(vector)
        self.messages_list.append(message)


def test_text_model():
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    esm = EntitySimilarityModel()
    esm.train(messages)
    new_msg = Message("JPS? idea SomeUnknownEntity", '', '', '')
    esm.update_model(new_msg)
    print(esm.find_similars(new_msg)[0][1])

if __name__ == '__main__':
    test_text_model()
