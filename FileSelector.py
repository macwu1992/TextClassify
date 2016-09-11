# -*-coding: utf-8-*-
'''
由于笔记本的性能不够
所以对每个类的样本进行随机取样
'''


import os, shutil
import random
from GetFileList import getFileList

def DocSelect(doc_folder_path):
    doc_list = []
    doc_list = getFileList(doc_folder_path, doc_list)
    training_list = []
    predict_list = []
    for doc in doc_list:
        rand_num = random.randrange(1, 11)
        if rand_num > 1 and rand_num <= 3 and doc.split('.')[-1] == 'txt':
            training_list.append(doc)

        elif rand_num <= 1 and doc.split('.')[-1] == 'txt':
            predict_list.append(doc)

    return training_list, predict_list

def FileCopy(src_folder_path, dest_folder_path):

    training_list, predict_list = DocSelect(src_folder_path)

    for file_path in predict_list:
        class_name = os.path.split(file_path)[0].split('/')[7]
        file_name = os.path.split(file_path)[1]

        if 'segment' not in class_name:
            class_path = os.path.join(dest_folder_path, class_name)
            if not os.path.exists(class_path):
                os.mkdir(class_path)
            else:
                new_file_path = os.path.join(class_path, file_name)
                if not os.path.exists(new_file_path):
                    shutil.copy(file_path, new_file_path)


doc_folder_path =  '/Users/Tong/workspace/git repos/TextClassify/文章分类最终稿'

dest_folder_path = './mysample'


FileCopy(doc_folder_path, dest_folder_path)
