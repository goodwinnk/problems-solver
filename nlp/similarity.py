import json
import numpy as np
import scipy
import gensim
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from nlp.corpus import DummyStemmer
from nlp.message_processing import extract_tokens, parse_text


def normalize(text, stopwords=(), stemmer=None, use_quoted=True):
    stemmer = stemmer if stemmer else DummyStemmer()
    stemmed_stopwords = set(stemmer.stem(word) for word in stopwords)
    result = []
    for token in extract_tokens(parse_text(text), targets=['text', 'link', '????' if not use_quoted else 'quoted']):
        stemmed = stemmer.stem(token)
        if stemmed not in stemmed_stopwords:
            result.append(stemmed)
    return result


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
        result.append({"origin": topics[i], "similars": []})
        for sim, index in similars:
            result[-1]['similars'].append({"similarity": float(sim), "text": topics[index]})
    return result


def tfidf_results(topics, tfidf_matrix, filename):
    similarity = get_similarity_matrix(tfidf_matrix)
    similar_topics = build_similarities(topics, similarity)
    json.dump(similar_topics, open(f'data/vectorization/{filename}', 'w', encoding='utf-8'), ensure_ascii=False,
              indent=4)


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


def get_tfidf_matrix(corpus: list):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    return vectorizer.fit_transform(map(lambda x: ' '.join(x), corpus))


def train_doc2vec():
    all_topics = list(
        map(lambda x: x['text'], json.load(open('data/processed/all_messages_without_sys_and_unk.json'))))
    porter = PorterStemmer()
    my_stopwords = stopwords.words('english')
    train_corpus = list(
        gensim.models.doc2vec.TaggedDocument(normalize(x, my_stopwords, porter, use_quoted=False), [i])
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


def main():
    topics = list(map(lambda x: x['text'], json.load(open('data/processed/all_topics.json'))))
    porter = PorterStemmer()
    my_stopwords = stopwords.words('english')
    normalized = list(map(lambda x: normalize(x, my_stopwords, porter), topics))
    tfidf_matrix = get_tfidf_matrix(normalized)
    tsne_res = tsne_transform(tfidf_matrix)
    scatter_plot(tsne_res)
    # doc2vec_results(topics, normalized, 'doc2vec_window3_min_count2_unquoted.json')


if __name__ == "__main__":
    main()
