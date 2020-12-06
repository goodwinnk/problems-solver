import json
from random import shuffle
from nltk import PorterStemmer
from nlp.corpus import DummyStemmer
from nlp.message_processing import extract_tokens, parse_text, read_data
from sklearn.naive_bayes import BernoulliNB


def check_contain_words(parsed, stemmed_words: set, stemmer=None) -> list:
    stemmer = DummyStemmer() if stemmer is None else stemmer
    tokenized = set(map(lambda x: stemmer.stem(x), extract_tokens(parsed, True, targets=['text', 'link', 'quoted'])))
    return [int(word in tokenized) for word in stemmed_words]


def is_code_contain_problem(parsed):
    code_blocks = filter(lambda x: x['type'] == 'code' or x['type'] == 'quoted', parsed)
    error_messages = ['error', 'exception', 'warning', 'fail', 'resolv', 'could not']
    for block in code_blocks:
        for message in error_messages:
            if message in block['text'].lower():
                return True
    return False


# def get_features(message: dict):
#     parsed = parse_text(message['text'])
#     has_problem = 0 + any(
#         check_contain_words(parsed, {'error', 'issu', 'work', 'except', 'fail', 'not'}, stemmer=PorterStemmer()))
#     need_help = 0 + any(
#         check_contain_words(parsed, {'help', 'pleas', 'anyon', 'tell', 'question'}, stemmer=PorterStemmer()))
#     non_question_specific = 0 + any(
#         check_contain_words(parsed, {'commit', 'object', 'thread', 'discuss'}, stemmer=PorterStemmer()))
#     question_mark = 0 + any(map(lambda x: x['type'] != 'code' and '?' in x['text'], parsed))
#     contain_code = 0 + any(map(lambda x: x['type'] == 'code', parsed))
#     contain_linkS = sum(map(lambda x: x['type'] == 'link', parsed)) > 1
#     features = [has_problem, need_help, question_mark, contain_code, non_question_specific, contain_linkS]
#     return features


# def old():
#     data = read_data('data/tagged/tagged_topics.json')
#     scores = []
#     negative_score = []
#     for _ in range(10):
#         shuffle(data)
#         threshold = int(len(data) * 0.8)
#         train, test = data[:threshold], data[threshold:]
#         trainX, trainY = list(map(get_features, train)), list(map(lambda x: x['is_question'], train))
#         testX, testY = list(map(get_features, test)), list(map(lambda x: x['is_question'], test))
#         gnb = BernoulliNB()
#         predY = gnb.fit(trainX, trainY).predict(testX)
#         scores.append(sum(a == b for a, b in zip(testY, predY)) / len(predY))
#         negatives = list(filter(lambda x: x, testY))
#         negative_score.append(sum(a == b and a == False for a, b in zip(testY, predY)) / len(negatives))
#     print('Total: ', sum(scores) / len(scores))
#     print('Negatives: ', sum(negative_score) / len(negative_score))


def is_question(message) -> bool:
    parsed = parse_text(message['text'])
    problem_in_code = is_code_contain_problem(parsed)
    contain_problem = any(  # broke, resolve???
        check_contain_words(parsed, {'error', 'issu', 'problem', 'except', 'wrong', 'fail', 'fall', 'crash', 'broke'},
                            PorterStemmer()))
    question_words = any(check_contain_words(parsed, {'why', 'how', 'help'}, PorterStemmer()))
    question_mark = any(map(lambda x: x['type'] != 'code' and '?' in x['text'], parsed))
    return contain_problem or problem_in_code or question_mark or question_words


if __name__ == '__main__':
    data = read_data('data/tagged/tagged_topics.json')
    positive, negative = [], []
    for message in data:
        if is_question(message):
            positive.append(message)
        else:
            negative.append(message)
    print(len(positive), len(negative))
    json.dump(positive, open('data/tagged/new_positive.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(negative, open('data/tagged/new_negative.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
