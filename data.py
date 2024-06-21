import os,sys
from os.path import join,exists
import random

class TrainValDataset:
    def __init__(self,dataset,k=5):
        self.k = k
        self.dataset = dataset
        self.genuine_set = [join(dataset,'0_Genuine',i) for i in os.listdir(join(dataset,'0_Genuine'))]
        self.forgeries_set = [join(dataset,'1_Forgeries',i) for i in os.listdir(join(dataset,'1_Forgeries'))]

        random.shuffle(self.genuine_set)
        random.shuffle(self.forgeries_set)

        self.label_set = dict()

        for i in self.genuine_set:
            self.label_set[i] = 0 # 0表示正样本
        
        for i in self.forgeries_set:
            self.label_set[i] = 1 # 1表示负样本

        # self.k_genuine是一个维度为(k,num//k)的列表。其中self.k_genuine[i]表示第i组真实数据
        self.k_genuine, self.k_forgeries = self.split_k(k)

    # 划分为k折
    def split_k(self,k):
        genuine_len = len(self.genuine_set) // k
        forgeries_len = len(self.forgeries_set) // k

        k_genuine = []
        k_forgeries = []
        for i in range(k):
            k_genuine.append(self.genuine_set[i*genuine_len:(i+1)*genuine_len])
            k_forgeries.append(self.forgeries_set[i*forgeries_len:(i+1)*forgeries_len])
        
        return k_genuine,k_forgeries


    # 将第idx组作为测试数据，其余组作为训练数据返回
    def getdata(self,idx):
        if idx >= self.k:
            raise ValueError(f'Index must be in range of 0~{self.k-1}')
            

        trainset = []
        trainlabelset = []
        valset = []
        vallabelset = []

        for i in range(self.k):
            if i == idx:
                valset = self.k_genuine[i] + self.k_forgeries[i]
            else:
                trainset = trainset + self.k_genuine[i] + self.k_forgeries[i]

        random.shuffle(trainset)
        random.shuffle(valset)

        for i in range(len(trainset)):
            trainlabelset.append(self.label_set[trainset[i]])

        for i in range(len(valset)):
            vallabelset.append(self.label_set[valset[i]])



        return trainset,trainlabelset,valset,vallabelset








if __name__ =='__main__':
    a = TrainValDataset('datasets\SigComp',5)

    trainset,trainlabelset,valset,vallabelset = a.getdata(0)

    print(trainset)
    print(len(trainset))
    print(len(trainlabelset))
    print(len(valset))
    print(len(vallabelset))