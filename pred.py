#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-5-19 下午11:09
# @Author  : lvconl
# @File    : pred.py
# @Software: PyCharm

from .cln_data import proc_text,extract_feat_from_data

dataset_path = './dataset'
pred_out_put = 'pred.txt'

context = '1）大专以上学历，应往届理工类毕业生(在读学生需提供学生证)，致力于向IT互联网朝阳行业发展，有志追求高薪，高品质生活者;并愿意在北京工作；  2）计算机（网络)、电子信息、软件工程、（电气）自动化、测控、生仪、机电等专业优先考虑；3）有计算机语言基础者优先，如：C、C++、C#、JAVA'

