# Part 1 - 데이터 전처리
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# XGBoost에 적용하기 위해 데이터를 DMatrix 형식으로 변환
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Part 2 - 모델 생성
params = {
    'objective' : 'reg:squarederror',   # 손실 함수
    'colsample_bytree': 0.3,    # 무작위로 선택할 특징(변수) 비율 설정 , 전체 특징중 30%만을 샘플링
    'learning_rate': 0.1,   # 학습률
    'max_depth': 5,     # 트리 최대 깊이
    'alpha' : 10    # L1 정규화 항, 모델의 복잡성을 줄여 과적합 방지
}

# 모델 학습
model = xgb.train(params, dtrain, num_boost_round=100)

# Part 3 - 예측
predictions = model.predict(dtest)

# 평균제곱오차(MSE)를 사용하여 모델 평가
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Part 4 - 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='red', label='Real Stock Price')
plt.plot(predictions, color='blue', label = 'Predicted Stock Price')
plt.title('Google Stock Price Prediction using XGBoost')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()