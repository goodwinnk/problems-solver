import json
from collections import defaultdict
from pprint import pprint
from typing import List
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from nlp.code_model import ErrorCodeSimilarityModel
from nlp.message_processing import Message, read_data
from nlp.text_model import TextSimilarityModel
import matplotlib.pyplot as plt


class DummyClassifier():
    def fit(self, a, b, **kwargs):
        pass

    def predict(self, x):
        return [True] * len(x)


def plot_field(start_x, end_x, start_y, end_y, x, y, classifier):
    field_x, field_y, field_c = [], [], []
    for xx in range(int(start_x * 100), int(end_x * 100 + 1), 1):
        for yy in range(int(start_y * 100), int(end_y * 100 + 1), 1):
            field_x.append(xx / 100)
            field_y.append(yy / 100)
            field_c.append('g' if classifier.predict([[field_x[-1], field_y[-1]]]) else 'b')
    plt.scatter(field_x, field_y, c=field_c)
    plot_features(x, y)
    plt.show()


def plot_features(x, y, show=False):
    x, y = zip(*list(sorted(zip(x, y), key=lambda v: v[1])))
    xs = tuple(zip(*x))
    plt.gca().set_facecolor('silver')
    plt.scatter(xs[0], xs[1], c=list(map(lambda decision: 'yellow' if decision else 'black', y)))
    if show:
        plt.show()


class DualModel:
    def __init__(self):
        self.text_model = TextSimilarityModel(n_components=100)
        self.code_model = ErrorCodeSimilarityModel(max_df=0.2)
        self.classifier = DecisionTreeClassifier(min_samples_leaf=5)

    def create_dataset(self, messages_list: List[Message], dataset_keys: List[List], plot=False):
        key_to_message = dict(map(lambda m: (m.get_key(), m), messages_list))
        x, y = [], []
        for record in dataset_keys:
            first, second, answer = key_to_message[record[0]], key_to_message[record[1]], record[2]
            if first == second:
                continue
            text_sim = self.text_model.compare_messages(first, second)
            code_sim = self.code_model.compare_messages(first, second)
            x.append([text_sim, code_sim])
            y.append(answer)
        if plot:
            plot_features(x, y, show=True)
        return x, y

    def test(self, messages_list: List[Message], dataset: List[List], do_train=True):
        if do_train:
            self.text_model.train(messages_list)
            self.code_model.train(messages_list)
        x, y = self.create_dataset(messages_list, dataset, True)
        train_x, train_y = x[:int(len(x) * 0.8)], y[:int(len(y) * 0.8)]
        test_x, test_y = x[int(len(x) * 0.8):], y[int(len(y) * 0.8):]
        if do_train:
            self.classifier.fit(train_x, train_y)
        else:
            test_x, test_y = x, y
        plot_field(-0.25, 1, 0, 1, x, y, self.classifier)
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
        self.text_model.train(messages_list)
        self.code_model.train(messages_list)
        x, y = self.create_dataset(messages_list, dataset)
        coef_true = len(y) / sum(y)
        weights = list(map(lambda s: coef_true if x else 1, y))
        self.classifier.fit(x, y, sample_weight=weights)

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
        return result

    def filter_candidates(self, candidates: dict, origin_msg):
        result = []
        for key, record in candidates.items():
            text_sim, code_sim = record['text_similarity'], record['code_similarity']
            if self.classifier.predict([[text_sim, code_sim]]):
                result.append(record['message'])
            elif text_sim > 0.75:
                print('_______________________________________________')
                print(origin_msg)
                print(record['message'])
        return result

    def get_similar_messages(self, message: Message):
        candidates = self.get_similar_candidates(message)
        result = self.filter_candidates(candidates, message)
        return result


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
    dataset = read_data('data/dataset/light_prod.json')
    shuffle(dataset)
    model = DualModel()
    model.train(messages, dataset)
    model.test(messages, dataset, do_train=False)
    positives = read_data('data/dataset/positives/all_positives.json')
    total, unknown_counter, has_counter = 0, 0, 0
    candidates = []
    for message in messages:
        result = model.get_similar_messages(message)
        total += len(result)
        for sim_message in result:
            if sim_message != message:
                record = [sim_message.get_key(), message.get_key(), True]
                if record not in positives:
                    unknown_counter += 1
                    candidates.append(record)
                else:
                    has_counter += 1
    json.dump(candidates, open('data/dataset/unknown.json', 'w'))
    print(f'Messages: ', len(messages))
    print(f'Found answers: ', total)
    print(f'Found new answers', unknown_counter)
    print(f'Matched with dataset: ', has_counter)

if __name__ == '__main__':
    test_text_model()
    # manual_filtering()
