#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午9:52
# @Author  : lvconl
# @File    : cln_data.py
# @Software: PyCharm
import os
import re
import jieba
import pandas as pd
import math
import nltk
import numpy as np
from nltk import TextCollection
from sklearn.naive_bayes import GaussianNB

dataset_path = './dataset'

text_filename = 'train_set.txt'

output_text_filename = 'raw_salary_text.csv'

output_cln_text_filename = 'clean_salary_text.csv'

text_w_salary_df_lst = []

test_file = os.path.join(dataset_path,text_filename)
def read_to_save():
    with open(test_file,'r',encoding='utf-8') as f:
        lines = f.read().splitlines()

    salarys = []
    for item in lines:
        line = item.split(',')
        salarys.append(line[0])

    test_series = pd.Series(lines)
    salarys_series = pd.Series(salarys)

    text_w_salary_df = pd.concat([salarys_series,test_series],axis=1)
    text_w_salary_df_lst.append(text_w_salary_df)

    result_df = pd.concat(text_w_salary_df_lst,axis=0)

    result_df.columns = ['salary','text']
    result_df.to_csv(os.path.join(dataset_path,output_text_filename),index = None,encoding = 'utf-8')

def proc_text(raw_line):
    stopwords = [line.rstrip() for line in open('stopwords.txt','r',encoding='utf-8')]
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    cln_line = filter_pattern.sub('',raw_line)

    word_lst = jieba.cut(cln_line)

    meaninful_words = []
    for word in word_lst:
        if word not in stopwords:
            meaninful_words.append(word)

    return ' '.join('%s'%word for word in meaninful_words)

def run_main():
    text_df = pd.read_csv(os.path.join(dataset_path,output_text_filename),encoding='utf-8')
    text_df['text'] = text_df['text'].apply(proc_text)

    text_df = text_df[text_df['text']!=' ']

    text_df.to_csv(os.path.join(dataset_path,output_cln_text_filename),index=None,encoding='utf-8')

    print('完成，并保存结果。')

def load_train_set():
    cln_text_df = pd.read_csv(os.path.join(dataset_path,output_cln_text_filename),encoding='utf-8')
    train_text_df = pd.DataFrame()
    test_text_df = pd.DataFrame()

    text_df = cln_text_df.reset_index()
    n_lines = text_df.shape[0]
    split_line_no = math.floor(n_lines*0.8)
    text_df_train = text_df.iloc[:split_line_no,:]
    text_df_test = text_df.iloc[split_line_no:,:]

    train_text_df = train_text_df.append(text_df_train)
    test_text_df = test_text_df.append(text_df_test)

    train_text_df = train_text_df.reset_index()
    test_text_df = test_text_df.reset_index()

    return train_text_df,test_text_df


def get_word_list_from_data(text_df):

    word_list = []
    for _,r_data in text_df.iterrows():
        word_list += r_data['text'].split(' ')

    return word_list

def extract_feat_from_data(text_df,text_collection,common_words_freqs):

    n_sample = text_df.shape[0]
    n_feat = len(common_words_freqs)
    common_words = [word for word,_ in common_words_freqs]

    X = np.zeros([n_sample,n_feat])
    y = np.zeros(n_sample)

    print('提取特征...')
    for i,r_data in text_df.iterrows():
        print('已完成{}个样本的特征提取'.format(i+1))

        text = r_data['text']
        feat_vec = []
        for word in common_words:
            if word in text:
                tf_idf_val = text_collection.tf_idf(word,text)
            else:
                tf_idf_val = 0

            feat_vec.append(tf_idf_val)

        X[i,:] = np.array(feat_vec)

        y[i] = int(r_data['salary'].split('-')[0])

    return X,y

def cal_acc(true_salarys,pred_salarys):

     n_total = len(true_salarys)
     correct_list = [true_salarys[i] == pred_salarys[i] for i in range(n_total)]

     acc = sum(correct_list) / n_total
     return acc


if __name__ == '__main__':

    #加载
    #run_main()


    train_text_df,test_text_df = load_train_set()

    n_common_words = 200

    all_words_in_train = get_word_list_from_data(train_text_df)

    fdisk = nltk.FreqDist(all_words_in_train)
    common_words_freqs = fdisk.most_common(n_common_words)

    for word,count in common_words_freqs:
        print('{}:{}'.format(word,count))

    text_collection = TextCollection(train_text_df['text'].values.tolist())
    print('训练样本特征提取')
    train_X,train_y = extract_feat_from_data(train_text_df,text_collection,common_words_freqs)
    print('完成')

    print('测试样本特征提取')
    test_X,test_y = extract_feat_from_data(test_text_df,text_collection,common_words_freqs)
    print('完成')

    print('训练模型...')
    gnb = GaussianNB()
    gnb.fit(train_X,train_y)
    print('训练完成...')

    print('测试模型...')
    test_pred = gnb.predict(test_X)
    print(test_pred)
    print('测试完成')

    print('准确率:',cal_acc(test_y,test_pred))