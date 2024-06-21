from data import TrainValDataset
from extractor import FeatureExtractor
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from skimage.io import imread_collection
from skimage.transform import resize
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=’auto_deprecated’, 
# 					coef0=0.0, shrinking=True, probability=False, tol=0.001, 
# 					cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
# 					decision_function_shape=’ovr’, random_state=None)

def getsvm(args):
    print(f'Prepareing SVM models...')
    if args.autogamma:
        gamma = 'auto'
    else:
        gamma = args.setgamma
    
    if args.kernel == 'linear':
        model = svm.SVC(kernel=args.kernel,max_iter=args.max_iter,C=args.C)
    elif args.kernel == 'poly':
        model = svm.SVC(kernel=args.kernel,
                        max_iter=args.max_iter,
                        C=args.C,
                        degree=args.degree,
                        gamma=gamma,
                        coef0=args.coef0
                        )
    elif args.kernel == 'rbf':
        model = svm.SVC(kernel=args.kernel,
                        max_iter=args.max_iter,
                        C=args.C,
                        gamma=gamma
                        )  
    else:
        raise ValueError(f'kernel must be in [linear,poly,rbf]')
    
    return model


def trainval(args):
    svm_model = getsvm(args)
    train_val_dataset = TrainValDataset(args.dataset,args.k)
    feature_extractor = FeatureExtractor(args.batch_size,args.device)
    
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    print(f'Starting training and testing...')
    for i in range(args.k):
        trainset,trainlabelset,valset,vallabelset = train_val_dataset.getdata(i)
        
        train_feature, train_feature_labels = feature_extractor.generate_stream(trainset,trainlabelset)
        
        svm_model.fit(train_feature, train_feature_labels)
        
        val_feature, val_feature_labels = feature_extractor.generate_stream(valset,vallabelset)
        
        pred = svm_model.predict(val_feature)
        cm = confusion_matrix(pred,val_feature_labels)
        tn,fp,fn,tp = cm.ravel()
        TN += tn
        FP += fp
        FN += fn
        TP += tp
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        acc = (tp + tn) / (tp + fn + fp + tn)
        print(f'*'*5+f'第{i+1}轮验证结果'+'*'*5)
        print(f'Accuracy: {acc}')
        print(f'fnr: {fnr}')
        print(f'fpr: {fpr}')
        print('*'*15)

    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    print('*'*5+f'平均结果'+'*'*5)
    print(f'ACC: {ACC}')
    print(f'FNR: {FNR}')
    print(f'FPR: {FPR}')
    print('*'*15)
        
    
#sigma        setgamma
#0.25         1.59439e-4
#0.5          7.97193e-5
#1.0          3.98596e-5
#1.5          2.65731e-5
#1.75         2.27769e-5



if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='SVM Parser')
    parser.add_argument('--kernel', type=str, default='rbf',choices=['linear','poly','rbf'],help='核函数（线性、多项式、高斯）')
    parser.add_argument('--C',type=float,default=2.5,help='惩罚系数')
    parser.add_argument('--degree',type=int,default=6,help='多项式核函数的阶数，只对poly有用')
    parser.add_argument('--autogamma',type=bool,default=False,help='自动gamma，值为1/_n_features')
    parser.add_argument('--setgamma',type=float,default=2.27769e-5,help='手动gamma，仅在autogamma=False有效，仅对rbf，poly有效。1 / (n_features * X.var())')
    parser.add_argument('--coef0',type=float,default=0.,help='核函数中的独立项，只对poly有用')
    parser.add_argument('--max_iter',type=int,default=-1,help='最大迭代次数，默认值为-1，表示不设限制')
    parser.add_argument('--dataset',type=str,default='datasets/LittleSig',help='数据路径')
    parser.add_argument('--k',type=int,default=5,help='k折交叉验证')
    parser.add_argument('--batch_size',type=int,default=16,help='特征提取器的batch_size')
    parser.add_argument('--device',type=str,default='cpu',help='特征提取器使用的计算设备')
    
    args = parser.parse_args()  # 获取所有参数
    
    
    print(f'Experiment settings:')
    print(args)
    
    print('-'*20+'START'+'-'*20)
    
    trainval(args)