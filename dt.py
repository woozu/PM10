import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lr
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

tree_result = pd.read_csv('tree_result.csv', sep=',', encoding='utf-8', header=0, engine='python')

#X, Y 분할하기
Y = tree_result['Range']
X = tree_result.drop(['PM10','Date', 'Range'], axis=1, inplace=False)

#훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

#결정 트리 분류 분석 : 1)모델 생성
dt_HAR = DecisionTreeClassifier(random_state=156)

#결정 트리 분류 분석 : 2) 모델 훈련
dt_HAR.fit(X_train, Y_train)

#결정 트리 분류 분석 : 3) 평가 데이터에 예측 수행 -> 예측 결과로 Y_predict 구하기
Y_predict = dt_HAR.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)
#print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy))

#print('결정 트리의 현재 하이퍼 매개변수 : \n', dt_HAR.get_params())
hyper_param = dt_HAR.get_params()
params={
    'max_depth' : [2,4,6,8,10,12,16,20]
}

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy', cv=5, return_train_score=True)
grid_cv.fit(X_train, Y_train)
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']]

#print('최고 평균 정확도 :{0:.4f}, 최적 하이퍼 매개변수:{1}'.format(grid_cv.best_score_, grid_cv.best_params_))

best_dt_HAR = grid_cv.best_estimator_
best_Y_predict = best_dt_HAR.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)

#print('best 결정 트리 예측 정확도 : {0:.4f}'.format(best_accuracy))

X_test10 = X_test[:10]
sample = X.head(1)
t_predict = dt_HAR.predict(sample)
real = tree_result.loc[0, 'PM10']

#print("테스트 데이터 : ", sample)
#print("예측값 : ", t_predict[0])
#print("실제값 : ", real)

def sample():
    return sample
def predict():
    return t_predict[0]
def real():
    return real

def result_accuracy():
    result_data = accuracy
    return result_data

def result_best_scroe():
    result_data = grid_cv.best_score_
    return result_data

def result_best_params():
    return grid_cv.best_params_

def result_best_accuracy():
    return best_accuracy