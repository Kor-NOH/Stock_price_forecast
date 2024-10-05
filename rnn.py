# Part 1 - 데이터 전처리
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 훈련 데이터를 불러옴
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

#----------------------------------------------------------------------
# Part 2 - RNN 모델 생성 및 학습
# Sequential - 레이어를 쌓아가는 신경망 구조
# LSTM - 장기 시계열 데이터 처리를 위한 신경망 구조
# Dropout - 과적합 방지를 위한 레이어
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# 모델 초기화
regressor = Sequential()

# LSTM 첫 번째 레이어
# 50개의 유닛을 가진 LTMS 레이어 추가 후, return_sequences=True를 통해 다음에도 추가할 수 있도록 함
# input_shape는 입력 데이터의 형상 지정
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Dropout(20%)을 추가해 과적합 방지
regressor.add(Dropout(0.2))

# LSTM 2, 3, 4 번째 레이어
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# 출력 레이어 추가
regressor.add(Dense(units = 1))

# RNN 모델 컴파일
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 모델 학습
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)

#----------------------------------------------------------------------
# Part 3 - 결과 예측
# 테스트 주가 데이터를 가져옴
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# 학습 데이터와 테스트 데이터 합치기
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# 테스트 데이터 전 ~ 마지막 60일 데이터를 가져와 입력 데이터로 변환
# 훈련데이터 20 + 60일 데이터 = 80개
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# 1차원 배열을 2차원 배열로 변경 후, 미니맥스를 사용하여 다시 스케일링
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# 테스트 데이터 구조 생성
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 예측 수행
predicted_stock_price = regressor.predict(X_test)
# 미니맥스로 변경한 값을 실제 주가와 비교하기 위해 원래 단위로 변완
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
