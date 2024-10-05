# Part 1 - 데이터 전처리
import numpy as np
import pandas as pd

# 데이터셋 불러오기
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

# 데이터셋 합치기
data = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# 시계열 데이터 기반으로 입, 출력 생성
def create_dataset(data, time_step = 1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

X, y = create_dataset(data.values, time_step=60)