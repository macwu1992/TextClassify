# -*-coding: utf-8-*-

import os
import random
import datetime

import jieba
from jieba import analyse

from prettytable import PrettyTable
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from scipy.sparse import csr_matrix
import numpy as np


def MakeWordSet(word_file_path):
    word_set = set()
    with open(word_file_path, 'r') as f:
        for line in f.readlines():
            word = unicode(line.strip(), 'utf-8')
            if len(word) > 0:
                word_set.add(word)
    return word_set

def TitleProcessing(predict_file_list):
    for file_name in predict_file_list:
        file_name = unicode(file_name, 'utf-8')
        print file_name.split(unicode('】', 'utf-8'))[0]


def TextProcessing(folder_path, test_size = 0.3):
    folder_list = os.listdir(folder_path)
    class_list = []
    data_list = []
    file_list = []

    jieba.analyse.set_stop_words('./stopwords_cn.txt')

    # 对于每个类循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        if not os.path.isdir(new_folder_path):
            continue
        files = os.listdir(new_folder_path)

        # 对于每个类中的文件循环,且每个类的文件不超过100个
        j = 1
        for file in files:
            file_path = os.path.join(new_folder_path, file)
            if j > 200:
                break
            with open(file_path, 'r') as f:
                raw = f.read()
            # word_list = list(jieba.cut(raw, cut_all=False))
            word_wright_list = jieba.analyse.extract_tags(raw, topK=100, withWeight=True)

            data_list.append(word_wright_list)
            class_list.append(unicode(folder, 'utf-8'))
            file_list.append(file)

            j += 1

    # 划分训练集与测试集
    class_data_list = zip(file_list, class_list, data_list)

    # 先将整个集合随机摆放,如果没有这一步,几乎就是无法识别
    random.shuffle(class_data_list)
    index = int(len(class_data_list)*test_size) + 1

    train_list = class_data_list[index:]
    predict_list = class_data_list[:index]

    train_file_list, train_class_list, train_data_list = zip(*train_list)
    predict_file_list, predict_class_list, predict_data_list = zip(*predict_list)

    all_words_dict = {}

    # 对于训练集中的词,加入词典,并计算词频
    for train_data in train_data_list:
        for data in train_data:
            if all_words_dict.has_key(data[0]):
                all_words_dict[data[0]] += data[1]
            else:
                all_words_dict[data[0]] = data[1]

    all_word_dict_sorted = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True)
    all_words_list = list(zip(*all_word_dict_sorted)[0])

    return all_words_list, predict_file_list, predict_class_list, predict_data_list, train_file_list, train_class_list, train_data_list

def words_dict(all_words_list, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    # n = 1
    for t in range(0, len(all_words_list), 1):
        # if n > 3000: # feature_words的维度不超过3000
        #     break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<all_words_list[t]:
            feature_words.append(all_words_list[t])
            # n += 1

    return feature_words

def TextFeatures(train_data_list, predict_data_list, feature_words):
    def text_features(data, feature_words):
        # sklearn特征 list
        # features = [1 if word in text_words else 0 for word in feature_words]
        features = [0]*len(feature_words)

        data_dict = {}
        for dat in data:
            data_dict[dat[0]] = dat[1]

        for word in feature_words:
            if data_dict.has_key(word):
                features[feature_words.index(word)] += data_dict[word]
        return features

    train_feature_list = [text_features(data, feature_words) for data in train_data_list]
    predict_feature_list = [text_features(data, feature_words) for data in predict_data_list]
    return train_feature_list, predict_feature_list


def TextClassifier(train_class_list, train_feature_list, predict_class_list, predict_feature_list):
    # 为训练集生成csr_matrix
    train_csr = csr_matrix(np.array(train_feature_list), dtype=np.float64)

    # 为预测集生成csr_matrix
    predict_csr = csr_matrix(np.array(predict_feature_list), dtype=np.float64)

    # classifier_SVC = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)).fit(train_csr, train_class_list)
    classifier_SVC = svm.LinearSVC().fit(train_csr, train_class_list)
    classifier_Bayes = MultinomialNB().fit(train_csr, train_class_list)

    predict_class_SVC = classifier_SVC.predict(predict_csr)
    predict_class_Bayes = classifier_Bayes.predict(predict_csr)

    predict_proba_SVC = classifier_SVC.decision_function(predict_csr)
    predict_proba_Bayes = classifier_Bayes.predict_proba(predict_csr)

    return predict_class_SVC, predict_proba_SVC, predict_class_Bayes, predict_proba_Bayes

def ResultWriter(predict_file_list, result_file_path, predict_class_list, classifier_class, predict_proba):
    if not os.path.exists(result_file_path):
        result_file = open(result_file_path, 'w')

        row = PrettyTable()
        row.field_names = ['文件名', '原类名', '预测类名', '类别1', '概率1', '类别2', '概率2']

        for i in range(len(predict_file_list)):
            sorted_predict_proba = sorted(predict_proba[i], reverse=True)

            max_proba_index = list(predict_proba[i]).index(sorted_predict_proba[0])
            second_proba_index = list(predict_proba[i]).index(sorted_predict_proba[1])

            row.add_row([predict_file_list[i], predict_class_list[i], classifier_class[i],
                         classifier_class[max_proba_index], sorted_predict_proba[0], classifier_class[second_proba_index], sorted_predict_proba[1]])

        result_file.write(row.get_string().encode('utf-8'))
        result_file.write(metrics.classification_report(predict_class_list, classifier_class).encode('utf-8'))

        result_file.close()

def ResultGenerator(predict_file_list, predict_class_list, classifier_class_list, predict_proba_list):
    result_folder_path = os.path.join('./','result')
    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)

    result_file_path_list = []

    result_file_path_SVC = os.path.join(result_folder_path, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + 'SVC_result.txt')
    result_file_path_Bayes = os.path.join(result_folder_path, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + 'Bayes_result.txt')

    result_file_path_list.append(result_file_path_SVC)
    result_file_path_list.append(result_file_path_Bayes)

    for i in range(len(result_file_path_list)):
        ResultWriter(predict_file_list, result_file_path_list[i], predict_class_list, classifier_class_list[i], predict_proba_list[i])

if __name__ == '__main__':
    file_folder = './mysample'
    all_words_list, predict_file_list, predict_class_list, predict_data_list, train_file_list, train_class_list, train_data_list = TextProcessing(file_folder)

    stopwords_file_path = './stopwords_cn.txt'
    stopwords_set = MakeWordSet(stopwords_file_path)

    test_accuracy_list = []
    feature_words = words_dict(all_words_list, stopwords_set)

    train_feature_list, predict_feature_list = TextFeatures(train_data_list, predict_data_list, feature_words)
    predict_class_SVC, predict_proba_SVC, predict_class_Bayes, predict_proba_Bayes = TextClassifier(train_class_list, train_feature_list, predict_class_list, predict_feature_list)

    classifier_class_list = []
    classifier_class_list.append(predict_class_SVC)
    classifier_class_list.append(predict_class_Bayes)

    predict_proba_list = []
    predict_proba_list.append(predict_proba_SVC)
    predict_proba_list.append(predict_proba_Bayes)

    ResultGenerator(predict_file_list, predict_class_list, classifier_class_list, predict_proba_list)