# -*- coding: utf-8 -*-
"""
   Author :       kouyafei
   date：          2018/4/4
"""
from collections import Counter
import jieba
import math
import numpy as np
import random


class Word2Vec:

    def __init__(self,win_size,embedding_size,learning_rate,max_iter):
        self.TABLE_SIZE = 100
        self.win_size = win_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.word2index = dict()
        self.index2word = dict()
        self.table = list()
        self.word_num = 0

    @staticmethod
    def word_counter(text):
        cut_result = jieba.cut(text)
        cut_result = list(cut_result)
        word_freq = Counter(cut_result)
        return word_freq, cut_result

    def generate_context(self, text):
        context_data = []
        length = self.win_size / 2
        for index, word in enumerate(text):
            pair = list()
            pair.append(word)
            skip_gram = list()
            for i in range(length):
                pos = i + 1
                if index - pos >= 0:
                    skip_gram.append(text[index - pos])
                if index + pos < len(text):
                    skip_gram.append(text[index + pos])
            pair.append(skip_gram)
            context_data.append(pair)
        return context_data

    def data_process(self, text):
        word_freq, cut_result = self.word_counter(text)
        self.word2index = {v: k for k, v in enumerate(word_freq)}
        self.index2word = {v: k for k, v in self.word2index.iteritems()}
        self.table = self.negative_sample(word_freq)
        word_freq = word_freq.most_common()
        context_data = self.generate_context(cut_result)
        self.word_num = len(word_freq)
        one_hot_pairs = []
        for item in context_data:
            pair_data = list()
            word = [0 for _ in range(self.word_num)]
            word[self.word2index[item[0]]] = 1
            context_word = list()
            for contexts in item[1]:
                context = [0 for _ in range(self.word_num)]
                context[self.word2index[contexts]] = 1
                context_word.append(context)
            pair_data.append(word)
            pair_data.append(context_word)
            one_hot_pairs.append(pair_data)
        return one_hot_pairs

    def negative_sample(self, word_freq):
        normalization = 0
        start = 0
        end = 0
        index = 0
        word_sample_dict = dict()
        table = [0 for _ in range(self.TABLE_SIZE)]
        for key, val in word_freq.items():
            word_sample_dict[key] = math.pow(val, 0.75) * self.TABLE_SIZE
            normalization += math.pow(val, 0.75)
        for key, val in word_sample_dict.items():
            end += int(val / normalization)
            word_sample_dict[key] = end
            index = self.word2index[key]
            for i in range(start, end):
                table[i] = index
            start += int(val / normalization)
        if end != self.TABLE_SIZE:
            for i in range(end, self.TABLE_SIZE):
                table[i] = index
        return table

    @staticmethod
    def sigmoid(x):
        return 1.0/(1+np.exp(-x))

    def train(self, text):
        data = self.data_process(text)
        word2vec = np.random.random((self.word_num, self.embedding_size))
        theta = np.random.random((self.embedding_size, self.word_num))
        for _ in range(self.max_iter):
            for item in data:
                word = np.array(item[0])
                word_index = word.argmax()
                contexts = item[1]
                neg_num = len(contexts)
                k = 0
                neg_list = list()
                neg_list.append(word)
                while k < neg_num:
                    rand = random.randint(0,self.TABLE_SIZE-1)
                    sample_index = self.table[rand]
                    if sample_index != word_index:
                        neg = [0 for _ in range(self.word_num)]
                        neg[sample_index] = 1
                        neg = np.array(neg)
                        neg_list.append(neg)
                        k += 1
                for context in contexts:
                    error = [0 for _ in range(self.embedding_size)]
                    context = np.array(context)
                    context_index = context.argmax()
                    for i, u in enumerate(neg_list):
                        q = self.sigmoid(word2vec[context_index].dot(theta))
                        if i == 0:
                            l = word
                        else:
                            l = [0 for _ in range(self.word_num)]
                        g = self.learning_rate*(l-q)
                        error += g.dot(theta.T)
                        theta += word2vec[context_index].reshape(self.embedding_size, 1).dot(g.reshape((1, self.word_num)))
                    word2vec[context_index] += error
        print word2vec


if __name__ == '__main__':
    pass
