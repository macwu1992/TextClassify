开发环境:
	系统：macOS Sierra 10.12
	IDE：Pycharm 社区版
	python 版本: 2.7

------------------------------------------------------------

使用包：
jieba, sklearn, numpy, prettytable

推荐使用virtualenv 创建虚拟环境

------------------------------------------------------------

主文件：
TextClassify.py
	包括jieba分词，SVM、Bayes分类器，以及分类结果的输出
	分类结果的输出路径为'./result/xxx.txt'，以运行时间为结果文件名。

工具类：
GetFileList.py
	获取数据集中的文件路径表

FileSelector.py
	由于笔记本性能原因，跑不了整个数据集，故编写工具类对数据进行随机取样。

stopwords_cn.txt为停用词表