import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

result = pd.read_csv('result.csv', sep=',', encoding='utf-8', header=0, engine='python')

#null값 채우기/ 강수량은 0, 나머지 값은 중앙값으로
result['Max_temp'] = result['Max_temp'].fillna(result['Max_temp'].median())
result['Rain'] = result['Rain'].fillna(0)
result['Wind_speed'] = result['Wind_speed'].fillna(result['Wind_speed'].median())
result['Wind_dir'] = result['Wind_dir'].fillna(result['Wind_dir'].median())
result.isnull().sum()

#선형 회귀 분석 모델 구축
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#X, Y 분할하기
Y = result['PM10']
X = result.drop(['PM10','Date'], axis=1, inplace=False)

#훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#선형 회귀 분석 : 모델 생성
lr = LinearRegression()

#선형 회귀 분석 : 모델 훈련
lr.fit(X_train, Y_train)

#선형 회귀 분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr.predict(X_test)

X_test10 = X_test[:10]
sample = X.head(1)
t_predict = lr.predict(sample)
real = result.loc[0, 'PM10']
score = r2_score(Y_test, Y_predict)

print("테스트 데이터 : ", sample)
print("예측값 : ", t_predict[0])
print("실제값 : ", real)
print("오차값 : ", abs(t_predict[0]-real))

import numpy as np

#평가하기
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score): {0:.3f}'.format(r2_score(Y_test, Y_predict)))

print('Y 절편 값 : ', np.round(lr.intercept_, 2))
print('회귀 계수 값 : ', np.round(lr.coef_, 2))

X_test10 = X_test[:10]
sample = X.head(1)
t_predict = lr.predict(sample)
real = result.loc[0, 'PM10']
errorv = float(abs(t_predict[0]-real))

print("테스트 데이터 : ", sample)
print("예측값 : ", t_predict[0])
print("실제값 : ", real)
print("오차값 : ", errorv)


def sample():
    return sample
def predict():
    return t_predict[0]
def real():
    return real
def errorv():
    return errorv

def result():
    return score