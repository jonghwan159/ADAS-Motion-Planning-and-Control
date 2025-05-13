import numpy as np
import matplotlib.pyplot as plt
from lane_2 import lane
from ex01_PathPlanning_BothLane import Global2Local, Polyfit, VehicleModel_Lat, PurePursuit


class LaneWidthEstimator(object):
    def __init__(self, Lw_init=3.0):
        self.Lw = Lw_init

    def update(self, coeff_L, coeff_R, isLaneValid_L, isLaneValid_R):
        valid_left = True
        valid_right = True
        for i in range(len(isLaneValid_L)):
            valid_left = valid_left and isLaneValid_L[i]
            valid_right = valid_right and isLaneValid_R[i]
        if valid_left and valid_right:
            self.Lw = coeff_L[-1] - coeff_R[-1]


def EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, Lw):
    valid_left = True
    valid_right = True
    num_order = len(coeff_L)
    for i in range(len(isLaneValid_L)):
        valid_left = valid_left and isLaneValid_L[i]
        valid_right = valid_right and isLaneValid_R[i]

    coeff_path_L = []
    coeff_path_R = []
    if valid_left and valid_right:
        for i in range(num_order):
            coeff_path_L.append(coeff_L[i])
            coeff_path_R.append(coeff_R[i])
    elif valid_left:
        for i in range(num_order):
            coeff_path_L.append(coeff_L[i])
            coeff_path_R.append(coeff_L[i])
        coeff_path_R[-1] = coeff_L[-1] - Lw
    elif valid_right:
        for i in range(num_order):
            coeff_path_L.append(coeff_R[i])
            coeff_path_R.append(coeff_R[i])
        coeff_path_L[-1] = coeff_R[-1] + Lw
    else:
        coeff_path_L = [0.0, 0.0, 0.0, 0.0]
        coeff_path_R = [0.0, 0.0, 0.0, 0.0]

    coeff_path = []
    for i in range(num_order):
        coeff_path.append((coeff_path_L[i] + coeff_path_R[i]) / 2)
    return coeff_path


if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_lane = np.arange(0.0, 100.0, 0.1)
    Y_lane_L, Y_lane_R, Valid_L, Valid_R = lane(X_lane)

    LaneWidth = LaneWidthEstimator()
    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PurePursuit()

    time = []
    X_ego = []
    Y_ego = []

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        # Lane Info
        X_ref = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_L, Y_ref_R, isLaneValid_L, isLaneValid_R = lane(X_ref)
        # Global points (front 5 meters from the ego vehicle)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()
        # Converted to local frame
        local_points_L = Global2Local(global_points_L, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        local_points_R = Global2Local(global_points_R, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        # 3th order fitting
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        # Lane to path
        LaneWidth.update(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R)
        coeff_path = EitherLane2Path(coeff_L, coeff_R, isLaneValid_L, isLaneValid_R, LaneWidth.Lw)

        # Controller input
        controller.ControllerInput(coeff_path, Vx)
        ego_vehicle.update(controller.u, Vx)

    plt.figure(1, figsize=(13, 2))
    plt.plot(X_lane, Y_lane_L, 'k--')
    plt.plot(X_lane, Y_lane_R, 'k--', label="Reference")
    plt.plot(X_ego, Y_ego, 'b.', label="Vehicle Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    #    plt.axis("best")
    plt.grid(True)
    plt.show()