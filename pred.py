#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-5-19 下午11:09
# @Author  : lvconl
# @File    : pred.py
# @Software: PyCharm

import os
import pandas as pd

dataset_path = './dataset'

text_filename = 'cln_train_set.txt'
#text_filename = 'train_set.txt'

output_text_filename = 'raw_salary_text.csv'

output_cln_text_filename = 'clean_salary_text.csv'

cln_text_df = pd.read_csv(os.path.join(dataset_path,output_cln_text_filename),encoding='utf-8')

if __name__ == '__main__':
    print(cln_text_df)