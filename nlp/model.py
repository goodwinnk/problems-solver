import json
import pickle
import logging

from collections import defaultdict
from pprint import pprint
from typing import List
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from nlp.text_model import TextSimilarityModel
from nlp.code_model import ErrorCodeSimilarityModel
from nlp.entity_model import EntitySimilarityModel
from nlp.message_processing import Message, read_data

import matplotlib.pyplot as plt


class DummyClassifier:
    def fit(self, a, b, **kwargs):
        pass

    def predict(self, x):
        return [True] * len(x)


def plot_field(start_x, end_x, start_y, end_y, x, y, classifier):
    field_x, field_y, field_z, field_c = [], [], [], []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for xx in range(int(start_x * 100), int(end_x * 100 + 1), 10):
        for yy in range(int(start_y * 100), int(end_y * 100 + 1), 10):
            for zz in range(0, 101, 10):
                field_x.append(xx / 100)
                field_y.append(yy / 100)
                field_z.append(zz / 100)
                field_c.append((0.1, 0.5, 0.1, 0.3) if classifier.predict([[field_x[-1], field_y[-1], field_z[-1]]]) \
                                   else 'r')
    ax.scatter(field_x, field_z, field_y, c=field_c)
    ax.set_xlabel('text similarity')
    ax.set_ylabel('entity similarity')
    ax.set_zlabel('code similarity')
    plot_features(x, y, ax)
    plt.show()


def plot_features(x, y, ax=None, show=False):
    x, y = zip(*list(sorted(zip(x, y), key=lambda v: v[1])))
    xs = tuple(zip(*x))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs[0], xs[2], xs[1], c=list(map(lambda decision: 'blue' if decision else 'black', y)))
    if show:
        plt.show()


class Model:
    def __init__(self):
        self.text_model = TextSimilarityModel(n_components=250)
        self.code_model = ErrorCodeSimilarityModel(max_df=0.08)  # max_df=0.014
        self.entity_model = EntitySimilarityModel()
        self.classifier = DecisionTreeClassifier(min_samples_leaf=10)
        self.x, self.y = None, None
        self.filepath = None
        # n_comp=250, min_samples=15, max_df 0.08 Pr: 69.6% Rc: 68.6%
        # n_comp=250, min_samples=15, max_df 0.08 Pr: 69.6% Rc: 68.6%
        # n_comp=250, min_samples=15, max_df 0.014 Pr: 63.8 % Rc: 73.5%
        # n_comp=100, min_samples=20, max_df 0.08 Pr: 75.8% Rc: 64.7
        # n_comp=100, min_samples=20, max_df 0.014,Pr: 64.5% Rc: 70.5%
        # n_comp=100, min_samples=40, max_df 0.014,Pr: Rc: 64.5% Rc: 70.5%

    @classmethod
    def load_model(cls, filepath):
        model = pickle.load(open(filepath, 'rb'))
        if isinstance(model, Model):
            logging.info('Model was loaded')
            logging.info(f'Model contain {len(model.text_model.messages_list)} messages')
            return model
        raise TypeError("Wrong model class")

    def save_model(self, filepath='model'):
        pickle.dump(self, open(filepath, 'wb'))

    def create_dataset(self, messages_list: List[Message], dataset_keys: List[List], plot=False):
        key_to_message = dict(map(lambda m: (m.get_key(), m), messages_list))
        x, y = [], []
        for record in dataset_keys:
            first, second, answer = key_to_message[record[0]], key_to_message[record[1]], record[2]
            if first == second:
                continue
            text_sim = self.text_model.compare_messages(first, second)
            code_sim = self.code_model.compare_messages(first, second)
            entity_sim = self.entity_model.compare_messages(first, second)
            x.append([text_sim, code_sim, entity_sim])
            y.append(answer)
        if plot:
            plot_features(x, y, show=True)
        return x, y

    def update_model(self, message: Message):
        self.text_model.update_model(message)
        self.code_model.update_model(message)
        self.entity_model.update_model(message)
        logging.info('Model was updated')

    def test(self, messages_list: List[Message], dataset: List[List], do_train=True):
        if do_train:
            self.text_model.train(messages_list)
            self.code_model.train(messages_list)
            self.entity_model.train(messages_list)
            self.x, self.y = self.create_dataset(messages_list, dataset, True)
        train_x, train_y = self.x[:int(len(self.x) * 0.8)], self.y[:int(len(self.y) * 0.8)]
        test_x, test_y = self.x[int(len(self.x) * 0.8):], self.y[int(len(self.y) * 0.8):]
        if do_train:
            self.classifier.fit(train_x, train_y)
        else:
            test_x, test_y = self.x, self.y
        plot_field(0, 1, 0, 1, self.x, self.y, self.classifier)
        test_res = self.classifier.predict(test_x)
        stats = defaultdict(int)
        for f, s in zip(test_res, test_y):
            stats[f'total_class_{s}'] += 1
            if f != s:
                stats['incorrect'] += 1
                stats[f'error_class_{s}'] += 1
            else:
                stats['correct'] += 1
        pprint(stats)

    def train(self, messages_list: List[Message], dataset: List[List]):
        logging.info('Train text model')
        self.text_model.train(messages_list)
        logging.info('Train code model')
        self.code_model.train(messages_list)
        logging.info('Train entity model')
        self.entity_model.train(messages_list)
        logging.info('Estimators trained')
        logging.info('Creating dataset')
        self.x, self.y = self.create_dataset(messages_list, dataset)
        positive_count = sum(self.y)
        coef_true = 1  # len(self.y) / (positive_count) if positive_count else 1
        weights = list(map(lambda s: coef_true if s else 1, self.y))
        logging.info(f'Train classifier on {len(self.x)} samples')
        if len(self.x) <= 10 or positive_count in [0, len(self.x)]:
            self.classifier = DummyClassifier()
            logging.warning("DATASET IS EMPTY, SO CLASSIFIER IS NOT WORKING(always answer YES, its similar)")
        self.classifier.fit(self.x, self.y, sample_weight=weights)
        logging.info('Trained')

    def get_similar_candidates(self, message: Message):
        result = defaultdict(dict)
        text_sim = self.text_model.find_similars(message, 5)
        code_sim = self.code_model.find_similars(message, 5)
        for similarity, msg in text_sim:
            target = result[(msg.channel_id, msg.ts)]
            target['message'] = msg
            target['text_similarity'] = similarity
        for similarity, msg in code_sim:
            target = result[(msg.channel_id, msg.ts)]
            target['message'] = msg
            target['code_similarity'] = similarity
        for key, record in result.items():
            if 'text_similarity' not in record:
                record['text_similarity'] = self.text_model.compare_messages(message, record['message'])
            elif 'code_similarity' not in record:
                record['code_similarity'] = self.code_model.compare_messages(message, record['message'])
            record['entity_similarity'] = self.entity_model.compare_messages(message, record['message'])
        return result

    def filter_candidates(self, candidates: dict):
        result = []
        for key, rec in candidates.items():
            text_sim, code_sim, entity_sim = rec['text_similarity'], rec['code_similarity'], rec['entity_similarity']
            if self.classifier.predict([[text_sim, code_sim, entity_sim]]):
                result.append((rec['message'], [text_sim, code_sim, entity_sim]))
        return result

    def get_similar_messages(self, message: Message):
        candidates = self.get_similar_candidates(message)
        return self.filter_candidates(candidates)


def manual_filtering():
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    key_to_message = dict(map(lambda m: (m.get_key(), m), messages))
    positives = read_data('data/dataset/positives/all_positives.json')
    negatives = []
    unknown = json.load(open('data/dataset/unknown.json', 'r'))
    for index, record in enumerate(unknown):
        sim_record = [record[1], record[0], record[2]]
        if record in positives or sim_record in positives:
            print('Have answer')
            pass
        elif record in negatives or sim_record in negatives:
            print('Have answer')
            pass
        else:
            first, second = record[:2]
            first, second = key_to_message[first], key_to_message[second]
            print(f'---------------------------------------{index}')
            print(first)
            print('------VVVVVVVVVVSSSSSSSSS------')
            print(second)
            answer = '?'
            while answer not in ['y', 'n']:
                answer = input()
            if answer == 'y':
                positives.append(record)
                positives.append(sim_record)
            else:
                negatives.append(record)
                negatives.append(sim_record)
        json.dump(positives, open('data/dataset/new_positive.json', 'w'))
        json.dump(negatives, open('data/dataset/new_negative.json', 'w'))


def test_text_model():
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    dataset = read_data('data/dataset/production.json')
    shuffle(dataset)
    model = Model()
    model.train(messages, dataset)
    model.test(messages, dataset, do_train=False)
    positives = read_data('data/dataset/positives/all_positives.json')
    total, unknown_counter, has_counter = 0, 0, 0
    candidates = []
    for message in messages:
        result = model.get_similar_messages(message)
        total += len(result)
        for sim_message, similarity in result:
            if sim_message.text != message.text:
                record = [sim_message.get_key(), message.get_key(), True]
                if record not in positives:
                    unknown_counter += 1
                    candidates.append(record)
                else:
                    has_counter += 1
    json.dump(candidates, open('data/dataset/unknown.json', 'w'))
    print(f'Messages: ', len(messages), ', positive pairs:', len(positives))
    print(f'Found answers: ', total - len(messages))
    print(f'In dataset: {has_counter}')
    print(f'Unknown: {unknown_counter}')
    print(f'Precision', has_counter / (has_counter + unknown_counter))
    print(f'Recall: ', has_counter / len(positives))


def new_test():
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    dataset = read_data('data/dataset/production.json')
    shuffle(dataset)
    model = Model()
    model.train(messages, dataset)
    model.test(messages, dataset, False)
    # model.save_model('./models/C01CBLSMX0V')
    # model = Model.load_model('model')
    # text = "When I incrementally compile stdlib, I sometimes get an error `has several compatible actual declarations in modules &lt;kotlin-stdlib&gt;, &lt;kotlin-stdlib&gt;` on many actuals:\n<https://scans.gradle.com/s/nwmexsnqsykoe/console-log#L267>\nrestarting doesn't help, only clean helps"
    # res = model.get_similar_messages(Message(text, '', '', ''))
    # for item in res[0]:
    #     print('______________________________')
    #     print(item)


if __name__ == '__main__':
    # new_test()
    test_text_model()
