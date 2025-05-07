import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_Measure_init, step_time=0.1, m=0.1, modelVariance=0.001, measureVariance=0.05, errorVariance_init=1.0):
        # 상태공간 모델: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k)
        self.A = 1.0                         # 상태 전이 행렬
        self.B = step_time / m              # 입력 계수 (mass 고려)
        self.C = 1.0                         # 출력 행렬
        self.Q = modelVariance              # 모델 잡음 분산 (작을수록 모델을 더 신뢰)
        self.R = measureVariance            # 측정 잡음 분산 (작을수록 센서를 더 신뢰)

        self.x_estimate = y_Measure_init    # 초기 추정값
        self.P_estimate = errorVariance_init  # 초기 오차 공분산

    def estimate(self, y_measure, input_u):
        # === Prediction ===
        x_predict = self.A * self.x_estimate + self.B * input_u
        P_predict = self.A * self.P_estimate * self.A + self.Q

        # === Kalman Gain ===
        K = P_predict * self.C / (self.C * P_predict * self.C + self.R)

        # === Update ===
        self.x_estimate = x_predict + K * (y_measure - self.C * x_predict)
        self.P_estimate = (1 - K * self.C) * P_predict
        

if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example06.csv")

    y_estimate = KalmanFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i],signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



