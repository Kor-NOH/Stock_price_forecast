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

# 전체 데이터셋을 다시 분할 (학습 8: 테스트 2)
from sklearn.model_selection import train_test_split
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# XGBoost에 적용하기 위해 데이터를 DMatrix 형식으로 변환
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Part 2 - 모델 생성
params = {
    'objective' : 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha' : 10
}

# 모델 학습
model = xgb.train(params, dtrain, num_boost_round=100)
