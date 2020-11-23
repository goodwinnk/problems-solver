from nltk.stem import StemmerI
from .message_processing import frequency_analysis


class DummyStemmer(StemmerI):
    def stem(self, token):
        return token


def read_corpus(path) -> list:
    file = open(path, 'r')
    data = file.read().split('\n')
    result = []
    for item in data:
        item = item.split('\t')
        if len(item) == 2:
            result.append((item[1], int(item[0])))
    del data
    return result


def create_corpus(messages, stemmer: StemmerI, stopwords=(), merge=True) -> list:
    if merge:
        stemmer = DummyStemmer() if not stemmer else stemmer
        stats = frequency_analysis(messages, stemmer, set(stopwords))
        return sorted(list((key, value) for key, value in stats.items()), key=lambda x: x[1], reverse=True)
    return list(frequency_analysis([message], stemmer, set(stopwords)) for message in messages)


def save_corpus(corpus: list, path: str):
    with open(path, 'w') as file:
        file.write('\n'.join([f"{count}\t{token}" for token, count in corpus]))


def get_corpus_diff(first_corpus: list, second_corpus: list, freq_coef, lower=True) -> list:
    first_dict = dict(map(lambda x: (x[0].lower(), x[1]), first_corpus))
    second_dict = dict(map(lambda x: (x[0].lower(), x[1]), second_corpus))
    first_total = sum(map(lambda x: x[1], first_corpus))
    second_total = sum(map(lambda x: x[1], second_corpus))
    slang_tokens = set()

    for f_token, f_token_count in first_dict.items():
        if f_token_count / first_total > freq_coef * second_dict.get(f_token.lower(), 0) / second_total:
            slang_tokens.add(f_token)
    return list(filter(lambda x: x[0] in slang_tokens, first_corpus))
