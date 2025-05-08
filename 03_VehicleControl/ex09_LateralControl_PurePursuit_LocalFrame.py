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

    class PurePursuit(object):
        def __init__(self, step_time, coeff, Vx, L=2.7, lookahead=2.0):
            self.dt = step_time
            self.L = L
            self.lookahead = lookahead
            self.u = 0.0
            self.Vx = Vx

        def ControllerInput(self, coeff, Vx):
            # y = ax^3 + bx^2 + cx + d
            # 목표점은 lookahead 거리만큼 앞
            x_ld = self.lookahead
            a = coeff[0][0]
            b = coeff[1][0]
            c = coeff[2][0]
            d = coeff[3][0]
            y_ld = a*x_ld**3 + b*x_ld**2 + c*x_ld + d

            # 목표점과의 거리
            ld = np.sqrt(x_ld**2 + y_ld**2)

            # 조향각 계산
            alpha = np.arctan2(y_ld, x_ld)  # 로컬 좌표 기준
            delta = np.arctan2(2 * self.L * np.sin(alpha), ld)

            # 제한
            self.u = np.clip(delta, -0.52, 0.52)
    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    controller = PurePursuit(step_time, polynomialfit.coeff, Vx)
    
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


