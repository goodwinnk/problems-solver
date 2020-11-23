import json
from collections import defaultdict
from pprint import pprint

import numpy as np
import scipy
import gensim
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def tsne_transform(matrix, n=2):
    tsne = TSNE(n_components=n, metric='cosine')
    return tsne.fit_transform(matrix)


def scatter_plot(transformed):
    plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.show()


# def map_data_to_clusters(data, cluster_matching):
#     result = defaultdict(list)
#     if isinstance(cluster_matching, np.ndarray):
#         cluster_matching = cluster_matching.tolist()
#     for item, cluster in zip(data, cluster_matching):
#         result[cluster].append(item)
#     return result


def get_similarity_matrix(matrix, metric='cosine'):
    assert metric in ['euclidean', 'cosine']
    if isinstance(matrix, scipy.sparse.csr.csr_matrix) and metric == 'euclidean':
        matrix = matrix.todense()
    if metric == 'euclidean':
        metric_func = lambda x, y: 1 / sum((x[i] - y[i]) ** 2 for i in range(len(x)))
        length = matrix.shape[0]
        result = np.ndarray(shape=(length, length))
        for i in range(length):
            for j in range(i, length):
                result[i, j] = metric_func(matrix[i], matrix[j])
        return result
    else:
        return cosine_similarity(matrix, matrix)


def find_most_similar(origin_i, matrix, samples_count):
    result = []
    for i, sim in enumerate(matrix[origin_i, :]):
        if len(result) < samples_count and i != origin_i:
            result.append((sim, i))
            result.sort()
        elif i != origin_i and sim > result[0][0]:
            result.pop(0)
            result.append((sim, i))
            result.sort()
    result.reverse()
    return result


def build_similarities(topics, similarity_matrix, count=3):
    result = list()
    for i in range(len(topics)):
        similars = find_most_similar(i, similarity_matrix, count)
        result.append({"origin": topics[i], "origin_id": i, "similars": []})
        for sim, index in similars:
            result[-1]['similars'].append({"similarity": float(sim), "text": topics[index], "id": index})
    return result


def tfidf_results(topics, tfidf_matrix, filename=None):
    similarity = get_similarity_matrix(tfidf_matrix)
    similar_topics = build_similarities(topics, similarity)
    if filename is not None:
        json.dump(similar_topics, open(f'data/vectorization/{filename}', 'w', encoding='utf-8'), ensure_ascii=False,
                  indent=4)
    return similar_topics


def tfidf_tsne_results(topics, tfidf_matrix, filename):
    trans = tsne_transform(tfidf_matrix, n=2)
    similarity = get_similarity_matrix(trans, metric='euclidean')
    similar_topics = build_similarities(topics, similarity)
    json.dump(similar_topics, open(f'data/vectorization/{filename}', 'w', encoding='utf-8'), ensure_ascii=False,
              indent=4)


def tfidf_svd_results(topics, tfidf_matrix, filename):
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    svd_over_tfidf = svd.fit_transform(tfidf_matrix)
    print(svd_over_tfidf.shape)
    similarity = get_similarity_matrix(svd_over_tfidf)
    print(similarity.shape)
    similar_topics = build_similarities(topics, similarity)
    json.dump(similar_topics, open(f'data/vectorization/{filename}', 'w', encoding='utf-8'), ensure_ascii=False,
              indent=4)


def get_tfidf_matrix(corpus: list, min_df=0, max_df=1.0, ngram_range=(1, 3)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    return vectorizer.fit_transform(map(lambda x: ' '.join(x), corpus))


def train_doc2vec():
    all_topics = list(
        map(lambda x: x['text'], json.load(open('data/processed/all_messages_without_sys_and_unk.json'))))
    porter = PorterStemmer()
    my_stopwords = stopwords.words('english')
    train_corpus = list(
        gensim.models.doc2vec.TaggedDocument(normalize(x, my_stopwords, porter, use_quoted=True), [i])
        for i, x in enumerate(all_topics))
    # window = 2; min_count = 1 | 0: 4091, 1: 247, 2: 125, 3: 71
    # window = 3; min_count = 1 | 0: 4128, 1: 225, 2: 114, 3: 61,
    # window = 3; min_count = 2 | 0: 4276, 1: 187, 2: 101, 3: 58
    # window = 3; min_count = 2 | 0: 4176, 1: 224, 2: 95, 3: 62 | quotes not in use
    # window = 4; min_count = 1 | 0: 4158, 1: 215, 2: 112, 3: 54
    # window = 4; min_count = 2 | 0: 4257, 1: 189, 2: 97, 3: 71
    # window = 5; min_count = 1 | 0: 4134, 1: 208, 2: 102, 3: 61
    # window = 5; min_count = 2 | 0: 4267, 1: 195, 2: 105, 3: 48
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, window=3)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('data/models/doc2vec_window3_min_count2_unquoted')
    print('Trained and saved\nTesting...')
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        if doc_id % 100 == 0:
            print(f'Process: {doc_id}/{len(train_corpus)}')
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    import collections
    counter = collections.Counter(ranks)
    print(counter)
    return model


def doc2vec_results(topics, normalized, filename):
    try:
        model = gensim.models.Doc2Vec.load('doc2vec_window3_min_count2_unquoted')
    except FileNotFoundError as e:
        model = train_doc2vec()
    matrix = [model.infer_vector(normal_vec) for normal_vec in normalized]
    similarity = get_similarity_matrix(matrix)
    similar_topics = build_similarities(topics, similarity)
    json.dump(similar_topics, open(f'data/vectorization/doc2vec/{filename}', 'w', encoding='utf-8'), ensure_ascii=False,
              indent=4)


def all_results():
    topics = list(map(lambda x: x['text'], json.load(open('data/processed/all_topics.json'))))
    porter = PorterStemmer()
    my_stopwords = stopwords.words('english')
    normalized = list(map(lambda x: normalize(x, my_stopwords, porter), topics))
    tfidf_matrix = get_tfidf_matrix(normalized)
    tfidf_svd_results(topics, tfidf_matrix, 'tfidf+svd.json')
    tfidf_results(topics, tfidf_matrix, 'tfidf.json')
    doc2vec_results(topics, normalized, 'doc2vec_window3_min_count2_unquoted.json')


def build_histogram(filename):
    data = json.load(open(filename, 'r'))
    values = [item['similars'][0]['similarity'] for item in data]
    plt.hist(values, bins=50)
    plt.title(filename)
    plt.show()
    # tfidf cut 0.2
    # tfidf+svd cut 0.7


def manual_selector(cand, threshold, confirmed, item):
    answer = ''
    if cand['similarity'] > threshold:
        for conf_cand in confirmed[str(item['origin_id'])]['similars']:
            if conf_cand['id'] == cand['id']:
                answer = 'y'
                print('Found answer automatically')
                break
        if answer != 'y':
            print('\n\n\n\n')
            pprint(item['origin']['origin'])
            print("VVVVVVVVV_________SSSSSSSSSS")
            pprint(cand['text']['origin'])
            print('\n\n\n\n')
            while answer.lower() not in ['y', 'n']:
                answer = input()
    return answer


def threshold_selector(cand, threshold, confirmed, item):
    answer = 'n'
    if cand['similarity'] > threshold:
        answer = 'y'
    return answer


def selection(input_, threshold, output_fn=None, confirmed_fn=None, selector=manual_selector):
    data = input_
    if isinstance(input_, str):
        data = json.load(open(input_, 'r'))
    confirmed = defaultdict(lambda: {"origin": "", "similars": []})
    if confirmed_fn is not None:
        confirmed.update(json.load(open(confirmed_fn, 'r')))
    result = defaultdict(lambda: {"origin": "", "similars": []})
    for item in data:
        for cand in item['similars']:
            answer = selector(cand, threshold, confirmed, item)
            if answer == 'y':
                result[item['origin_id']]['similars'].append({'id': cand['id'], 'text': cand['text']})
                result[item['origin_id']]["origin"] = item['origin']
            if output_fn is not None:
                json.dump(result, open(output_fn, 'w'), ensure_ascii=False, indent=4)
    return result


def subtract_dataset(first_fn: str, second_fn: str):
    first = json.load(open(first_fn, 'r'))
    print(f'First size: ', len(first))
    second = json.load(open(second_fn, 'r'))
    print(f'Second size: ', len(second))
    result = defaultdict(lambda: {"origin": "", "similars": []})
    first_pairs = 0
    for key, value in first.items():
        first_pairs += len(value['similars'])
        if key in second:
            first_ids = list(map(lambda x: x['id'], first[key]['similars']))
            second_ids = list(map(lambda x: x['id'], second[key]['similars']))
            for item_id, item in zip(first_ids, first[key]['similars']):
                if item_id not in second_ids:
                    result[key]['origin'] = value['origin']
                    result[key]['similars'].append(item)
        else:
            result[key] = value
    print('First pairs: ', first_pairs)
    return result


def join_datasets(first_fn, second_fn):
    result = json.load(open(first_fn, 'r'))
    second = json.load(open(second_fn, 'r'))
    print(f'Provided:\n\tFirst: {len(result)}\n\tSecond: {len(second)}')
    for key, item in second.items():
        if key not in result:
            result[key] = item
        else:
            for sim_item in item['similars']:
                if sim_item not in result[key]['similars']:
                    result[key]['similars'].append(sim_item)
    print(f'Total: {len(result)}')
    return result


def f_score(true_answers: defaultdict, dataset: defaultdict, print_stats=False):
    true_positives, false_negatives, false_positives = 0, 0, 0
    for key, item in dataset.items():
        assert isinstance(key, str)
        for similar_item in item['similars']:
            if similar_item in true_answers[key]['similars']:
                true_positives += 1
            else:
                false_positives += 1
    for key, item in true_answers.items():
        assert isinstance(key, str)
        for similar_item in item['similars']:
            if similar_item not in dataset[key]['similars']:
                false_negatives += 1
                print('_____________________________')
                print(true_answers[key]['origin'], '\n\n')
                print(similar_item)
                print('\n\n')
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    if print_stats:
        print(f'True positives: {true_positives}')
        print(f'False positives: {false_positives}')
        print(f'False negatives: {false_negatives}')
        print(f'Precission {precision}')
        print(f'Recall {recall}')
        print(f'F-SCORE: {f_score}')
    return precision, recall, f_score


def main():
    true_answers = defaultdict(lambda: {"origin": "", "similars": []})
    true_answers.update(json.load(open('data/dataset/all.json', 'r', encoding='utf-8')))
    # dataset = defaultdict(lambda: {"origin": "", "similars": []})
    # dataset.update(json.load(open('data/dataset/tfidf+svd_0.7.json', 'r', encoding='utf-8')))
    # f_score(true_answers, dataset)

    dataset = defaultdict(lambda: {"origin": "", "similars": []})
    dataset.update(
        json.loads(json.dumps(selection('data/vectorization/tfidf+svd.json', 0.79, selector=threshold_selector))))
    f_score(true_answers, dataset, print_stats=True)
    precision, recall, f_scores, thresholds = [], [], [], []
    for threshold in range(60, 91):
        dataset = defaultdict(lambda: {"origin": "", "similars": []})
        dataset.update(
            json.loads(
                json.dumps(
                    selection('data/vectorization/tfidf+svd.json', threshold / 100, selector=threshold_selector))))
        pr, rc, fs = f_score(true_answers, dataset)
        precision.append(pr)
        recall.append(rc)
        f_scores.append(fs)
        thresholds.append(threshold)
    plt.plot(thresholds, precision, label='precision')
    plt.plot(thresholds, recall, label='recall')
    plt.plot(thresholds, f_scores, label='f_score')
    plt.legend()
    plt.show()

    # f_score('data/dataset/all.json', 'data/dataset/tfidf+svd_0.7.json')
    # pprint(subtract_dataset('data/dataset/tfidf+svd_0.7.json', 'data/dataset/tfidf_0.2.json'))
    # pprint(subtract_dataset('data/dataset/tfidf_0.2.json', 'data/dataset/tfidf+svd_0.7.json'))
    # build_histogram('data/vectorization/tfidf+svd.json')
    # json.dump(join_datasets('data/dataset/tfidf+svd_0.7.json', 'data/dataset/tfidf_0.2.json'),
    #           open('data/dataset/all.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
