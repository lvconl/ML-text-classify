#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午9:52
# @Author  : lvconl
# @File    : cln_data.py
# @Software: PyCharm
import os
import re
import jieba.posseg as pseg
import pandas as pd
import math
import nltk
from nltk import TextCollection

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
    filter_pattern = re.compile('^[\u4E00-\u9FA5A-Za-z]+$')
    cln_line = filter_pattern.sub('',raw_line)

    word_lst = pseg.cut(cln_line)

    meaninful_words = []
    for word in word_lst:
        if (word not in stopwords) and (word != ' '):
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

if __name__ == '__main__':
    train_text_df,test_text_df = load_train_set()

    n_common_words = 100

    all_words_in_train = get_word_list_from_data(train_text_df)

    fdisk = nltk.FreqDist(all_words_in_train)
    common_words_freq = fdisk.most_common(n_common_words)

    for word,count in common_words_freq:
        print('{}:{}'.format(word,count))