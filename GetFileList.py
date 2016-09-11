# -*-coding: utf-8-*-

import os


def getFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir) and dir.split('/')[-1] != '错':
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, fileList)
    return fileList