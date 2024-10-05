# Part 1 - 데이터 전처리
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 데이터셋을 불러옴
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# Open 컬럼만 선택
training_set = dataset_train.iloc[:, 1:2].values

# 데이터 스케일링 (미니맥스 사용)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 데이터를 60개씩 X 배열에 넣고, 그 다음 데이터(61번쨰)를 Y 배열에 넣음
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# LSTM 계층에 데이터를 입력하기위해 3차원 배열로 변환 (샘플 수, 타임스텝 수, 입력 차원 수)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
