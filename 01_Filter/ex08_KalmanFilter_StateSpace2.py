import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_measure_init, step_time=0.1, m=10.0, k=100.0, b=2.0, Q_x=0.01, Q_v=0.1, R=0.1, P_init=10.0):
        dt = step_time
        # 상태공간 모델
        self.A = np.array([
            [1.0, dt],
            [-k/m * dt, 1 - b/m * dt]
        ])
        self.B = np.array([
            [0.0],
            [dt/m]
        ])
        self.C = np.array([[1.0, 0.0]])

        # 공분산, 노이즈
        self.Q = np.array([
            [Q_x, 0.0],
            [0.0, Q_v]
        ])
        self.R = R
        self.x_estimate = np.array([[y_measure_init], [0.0]])  # [x, v]
        self.P_estimate = np.eye(2) * P_init

    def estimate(self, y_measure, input_u):
        # Prediction
        x_predict = self.A @ self.x_estimate + self.B * input_u
        P_predict = self.A @ self.P_estimate @ self.A.T + self.Q

        # Kalman Gain
        S = self.C @ P_predict @ self.C.T + self.R
        K = P_predict @ self.C.T @ np.linalg.inv(S)

        # Update
        y_predict = self.C @ x_predict
        self.x_estimate = x_predict + K * (y_measure - y_predict)
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_predict
if __name__ == "__main__":
    signal = pd.read_csv("01_filter/Data/example08.csv")

    y_estimate = KalmanFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i],signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate[0][0]

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



