from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

# --- PID Controller ---
class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=0.4, D_Gain=0.9, I_Gain=0.02):
        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain
        self.step_time = step_time
        self.prev_error = measure - reference
        self.integral = 0.0
        self.u = 0.0

    def ControllerInput(self, reference, measure):
        error = measure - reference
        self.integral += error * self.step_time
        derivative = (error - self.prev_error) / self.step_time
        self.u = -self.Kp * error - self.Kd * derivative - self.Ki * self.integral
        self.prev_error = error

# --- Low Pass Filter ---
class LowPassFilter:
    def __init__(self, y_initial_measure, alpha=0.1):
        self.y_estimate = y_initial_measure
        self.alpha = alpha

    def estimate(self, y_measure):
        self.y_estimate = (1 - self.alpha) * self.y_estimate + self.alpha * y_measure

# --- Kalman Filter ---
class KalmanFilter:
    def __init__(self, y_measure_init, step_time=0.1, m=1.0, Q_pos=0.01, Q_vel=0.1, R=0.5, P_init=1.0):
        dt = step_time
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.0], [dt/m]])
        self.C = np.array([[1.0, 0.0]])
        self.Q = np.array([[Q_pos, 0.0], [0.0, Q_vel]])
        self.R = R
        self.x_estimate = np.array([[y_measure_init], [0.0]])
        self.P_estimate = np.eye(2) * P_init

    def estimate(self, y_measure, input_u):
        x_predict = self.A @ self.x_estimate + self.B * input_u
        P_predict = self.A @ self.P_estimate @ self.A.T + self.Q
        S = self.C @ P_predict @ self.C.T + self.R
        K = P_predict @ self.C.T @ np.linalg.inv(S)
        y_predict = self.C @ x_predict
        self.x_estimate = x_predict + K @ (y_measure - y_predict)
        self.P_estimate = (np.eye(2) - K @ self.C) @ P_predict

# --- Main Simulation ---
if __name__ == "__main__":
    target_y = 0.0
    step_time = 0.1
    simulation_time = 30
    time = []
    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)

    # 결과 저장 리스트
    measure_y = []
    estimated_y_LPF = []
    estimated_y_KF = []

    # 각 필터, 컨트롤러
    lpf = LowPassFilter(plant.y_measure[0][0])
    kf = KalmanFilter(plant.y_measure[0][0])
    pid_LPF = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    pid_KF = PID_Controller(target_y, plant.y_measure[0][0], step_time)

    for i in range(int(simulation_time / step_time)):
        t = step_time * i
        time.append(t)
        current_measure = plant.y_measure[0][0]
        measure_y.append(current_measure)

        # LPF 추정 및 제어
        lpf.estimate(current_measure)
        estimated_y_LPF.append(lpf.y_estimate)
        pid_LPF.ControllerInput(target_y, lpf.y_estimate)

        # Kalman Filter 추정 및 제어
        kf.estimate(current_measure, pid_KF.u)
        estimated_y_KF.append(kf.x_estimate[0][0])
        pid_KF.ControllerInput(target_y, kf.x_estimate[0][0])

        # Vehicle에 Kalman Filter 기반 제어 입력 적용
        plant.ControlInput(pid_KF.u)

    # --- Plot ---
    plt.figure(figsize=(10,6))
    plt.plot([0, time[-1]], [target_y, target_y], 'k--', label="Reference (y=0)")
    plt.plot(time, measure_y, 'r:', label="Raw Measurement")
    plt.plot(time, estimated_y_LPF, 'b-', label="LPF Estimate")
    plt.plot(time, estimated_y_KF, 'g-', label="Kalman Estimate")
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicle y-position")
    plt.title("Comparison: Kalman Filter vs Low Pass Filter (with PID Control)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()