import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local
from ex06_GlobalFrame2LocalFrame import PolynomialFitting
from ex06_GlobalFrame2LocalFrame import PolynomialValue

    
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0-2*np.cos(X_ref/10)
    num_degree = 3
    num_point = 5
    x_local = np.arange(0.0, 10.0, 0.5)

    class PD_Controller(object):
        def __init__(self, step_time, Vx, Kp=1.5, Kd=0.2, L=2.7):
            self.step_time = step_time
            self.Kp = Kp
            self.Kd = Kd
            self.L = L      # 휠베이스
            self.Vx = Vx
            self.prev_error = 0.0
            self.u = 0.0    # 최종 조향각 결과

        def ControllerInput(self, coeff, Vx):
            # ------------------------------------
            # Feedforward term: 곡률 기반 조향각
            # 곡률 κ = (3a * x^2 + 2b * x + c) / (1 + (3ax^2 + 2bx + c)^2)^(3/2)
            # 여기서 x = 0이므로 d^2y/dx^2 = 2b, dy/dx = c
            a = coeff[0][0]
            b = coeff[1][0]
            c = coeff[2][0]
            # Feedforward 조향각: delta_ff = atan(L * curvature)
            curvature = 2 * b  # at x=0
            delta_ff = np.arctan(self.L * curvature)

            # ------------------------------------
            # Feedback term: PD 제어
            lateral_error = coeff[3][0]   # y(0) = d (오프셋 항)
            derivative = (lateral_error - self.prev_error) / self.step_time
            delta_fb = self.Kp * lateral_error + self.Kd * derivative
            self.prev_error = lateral_error

            # ------------------------------------
            # Total control
            self.u = delta_ff + delta_fb

            # 조향각 제한 (±30도 ≈ 0.52 rad)
            self.u = np.clip(self.u, -0.52, 0.52)
    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    controller = PD_Controller(step_time, polynomialfit.coeff, Vx)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_convert = 2.0-2*np.cos(X_ref_convert/10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)
        controller.ControllerInput(polynomialfit.coeff, Vx)
        ego_vehicle.update(controller.u, Vx)

        
    plt.figure(1)
    plt.plot(X_ref, Y_ref,'k-',label = "Reference")
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


