import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_3 import map

show_animation  = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.heading = 0.0
        self.f = 0
        self.g = 0
        self.h = 0

# Check if position of node is same( if distance < threshold, regard as same node)
def isSamePosition(node_1, node_2, epsilon_position=0.3):
    dx = node_1.position[0] - node_2.position[0]
    dy = node_1.position[1] - node_2.position[1]
    return math.hypot(dx, dy) < epsilon_position


def isSameYaw(node_1, node_2, epsilon_yaw=0.2):
    dyaw = abs(node_1.position[2] - node_2.position[2])
    dyaw = min(dyaw, 2 * math.pi - dyaw)  # wrap-around 처리
    return dyaw < epsilon_yaw


# Action set, Moving only forward direction              
def get_action(R,Vx,delta_time_step):
    yaw_rate = Vx/R
    distance_travel = Vx*delta_time_step
    # yaw_rate, delta_time_step, cost
    action_set = [[yaw_rate, delta_time_step, distance_travel], 
                  [-yaw_rate, delta_time_step, distance_travel],
                  [yaw_rate/2, delta_time_step, distance_travel],
                  [-yaw_rate/2, delta_time_step, distance_travel],
                  [0.0, delta_time_step, distance_travel]]
    return action_set

# Vehicle movement
def vehicle_move(position_parent, yaw_rate, delta_time, Vx):
    x_parent, y_parent, yaw_parent = position_parent

    if abs(yaw_rate) > 1e-5:
        R = Vx / yaw_rate
        cx = x_parent - R * math.sin(yaw_parent)
        cy = y_parent + R * math.cos(yaw_parent)
        yaw_child = yaw_parent + yaw_rate * delta_time
        x_child = cx + R * math.sin(yaw_child)
        y_child = cy - R * math.cos(yaw_child)
    else:
        # 직진
        x_child = x_parent + Vx * delta_time * math.cos(yaw_parent)
        y_child = y_parent + Vx * delta_time * math.sin(yaw_parent)
        yaw_child = yaw_parent

    # Normalize yaw
    yaw_child = yaw_child % (2 * math.pi)

    return [x_child, y_child, yaw_child]


# Collision check : path overlaps with any of obstacle
def collision_check(position_parent, yaw_rate, delta_time_step, obstacle_list, Vx):
    num_points = 10
    for i in range(1, num_points + 1):
        t = i * delta_time_step / num_points
        pos = vehicle_move(position_parent, yaw_rate, t, Vx)
        px, py = pos[0], pos[1]
        for obs in obstacle_list:
            ox, oy, r = obs
            if math.hypot(px - ox, py - oy) <= r:
                return True
    return False

# Check if the node is in the searching space
def isNotInSearchingSpace(position_child, space):
    x, y = position_child[0], position_child[1]
    x_min, x_max, y_min, y_max = space
    return not (x_min <= x <= x_max and y_min <= y <= y_max)
        
def heuristic(cur_node, goal_node):
    dist = np.sqrt((cur_node.position[0] - goal_node.position[0])**2 + (cur_node.position[1]  - goal_node.position[1])**2)
    return dist

def a_star(start, goal, space, obstacle_list, R, Vx, delta_time_step, weight):
    start_node = Node(None, start)
    goal_node = Node(None, goal)
    
    open_list = [start_node]
    closed_list = []

    while open_list:
        # 최소 f 값을 가지는 노드 선택
        current_index = np.argmin([node.f for node in open_list])
        cur_node = open_list.pop(current_index)
        closed_list.append(cur_node)

        # 목표 도달 여부 검사
        if isSamePosition(cur_node, goal_node, epsilon_position=0.4):
            print("Goal reached!")
            path = []
            while cur_node is not None:
                path.append(cur_node.position)
                cur_node = cur_node.parent
            return path[::-1]  # 경로를 역순으로 반환

        # 자식 노드 생성
        action_set = get_action(R, Vx, delta_time_step)
        for action in action_set:
            yaw_rate, delta_t, _ = action
            child_pos = vehicle_move(cur_node.position, yaw_rate, delta_t, Vx)

            if isNotInSearchingSpace(child_pos, space):
                continue
            if collision_check(cur_node.position, yaw_rate, delta_t, obstacle_list, Vx):
                continue

            child_node = Node(cur_node, child_pos)
            child_node.g = cur_node.g + Vx * delta_t
            child_node.h = heuristic(child_node, goal_node)
            child_node.f = child_node.g + weight * child_node.h

            # 이미 닫힌 리스트에 있는 경우 skip
            if any(isSamePosition(child_node, closed) and isSameYaw(child_node, closed) for closed in closed_list):
                continue

            # open_list에 동일 위치의 더 나은 노드가 있는 경우 skip
            skip = False
            for open_n in open_list:
                if isSamePosition(child_node, open_n) and isSameYaw(child_node, open_n):
                    if child_node.g >= open_n.g:
                        skip = True
                        break
            if skip:
                continue

            open_list.append(child_node)

        # 시각화
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.01)

    print("Path not found.")
    return []
                

def main():

    start, goal, obstacle_list, space = map()

    if show_animation == True:
        theta_plot = np.linspace(0,1,101) * np.pi * 2
        plt.figure(figsize=(8,8))
        plt.plot(start[0], start[1], 'bs',  markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs',  markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        for i in range(len(obstacle_list)):
            x_obstacle = obstacle_list[i][0] + obstacle_list[i][2] * np.cos(theta_plot)
            y_obstacle = obstacle_list[i][1] + obstacle_list[i][2] * np.sin(theta_plot)
            plt.plot(x_obstacle, y_obstacle,'k-')
        plt.axis(space)
        plt.grid(True)
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Hybrid a star algorithm", fontsize=20)

    opt_path = a_star(start, goal, space, obstacle_list, R=5.0, Vx=2.0, delta_time_step=0.5, weight=1.1)
    print("Optimal path found!")
    opt_path = np.array(opt_path)
    if show_animation == True:
        plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
        plt.show()


if __name__ == "__main__":
    main()

    

