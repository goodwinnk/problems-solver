# import json
# import re
# from collections import defaultdict
# from pprint import pprint
#
# import matplotlib.pyplot as plt
# from nlp.similarity import get_tfidf_matrix, tfidf_results, build_histogram, selection, threshold_selector, f_score, \
#     subtract_dataset
#
# from nlp.message_processing import parse_text, extract_tokens
#
#
#
#
#
#
#
#
# def process_data():
#     topics = json.load(open('data/processed/all_topics.json'))
#     code = extract_code(topics)
#     tfidf = get_tfidf_matrix(code)
#     return tfidf_results(topics, tfidf, 'code-tfidf.json')
#
#
# def f_score_plot():
#     true_answers = defaultdict(lambda: {"origin": "", "similars": []})
#     true_answers.update(json.load(open('data/dataset/code_0.2.json', 'r', encoding='utf-8')))
#     precision, recall, f_scores, thresholds = [], [], [], []
#     for threshold in range(10, 61):
#         dataset = defaultdict(lambda: {"origin": "", "similars": []})
#         dataset.update(
#             json.loads(
#                 json.dumps(
#                     selection('data/vectorization/code-tfidf_0.2.json', threshold / 100, selector=threshold_selector))))
#         pr, rc, fs = f_score(true_answers, dataset)
#         precision.append(pr)
#         recall.append(rc)
#         f_scores.append(fs)
#         thresholds.append(threshold)
#     plt.plot(thresholds, precision, label='precision')
#     plt.plot(thresholds, recall, label='recall')
#     plt.plot(thresholds, f_scores, label='f_score')
#     plt.legend()
#     plt.show()
#
#
# # def find_optimal_params(true_answers, topics, ):
#
#
# def main():
#     true_answers = defaultdict(lambda: {"origin": "", "similars": []})
#     true_answers.update(json.load(open('data/dataset/code_0.2.json', 'r', encoding='utf-8')))
#     topics = json.load(open('data/processed/all_topics.json'))
#     code = extract_code(topics)
#     # max_fs, thr, mxdf, mndf = 0, 0, 0, 0
#     # for min_df in range(1, 25):
#     #     for max_df in range(min_df+1, 25):
#     #         print(min_df, max_df, max_fs, thr, mxdf)
#     #         for threshold in range(10, 40):
#     #             tfidf = get_tfidf_matrix(code, max_df=max_df / 1000)
#     #             results = tfidf_results(topics, tfidf)
#     #             selected = defaultdict(lambda: {"origin": "", "similars": []})
#     #             selected.update(json.loads(json.dumps(selection(results, threshold / 100, selector=threshold_selector))))
#     #             pr, rec, fs = f_score(true_answers, selected)
#     #             if fs > max_fs:
#     #                 max_fs = fs
#     #                 thr = threshold
#     #                 mxdf = max_df
#     #                 mndf = min_df
#     #                 print(max_fs, thr, mxdf, mndf)
#     #                 f_score(true_answers, selected, True)
#
#     tfidf = get_tfidf_matrix(code, max_df=0.014)
#     results = tfidf_results(topics, tfidf)
#     selected = defaultdict(lambda: {"origin": "", "similars": []})
#     selected.update(json.loads(json.dumps(selection(results, 0.27, selector=threshold_selector))))
#     f_score(true_answers, selected, True)
#
#
# if __name__ == '__main__':
#     main()
