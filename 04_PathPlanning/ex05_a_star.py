import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
from map_2 import map

show_animation = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0  # total cost = g + h
        self.g = 0  # actual cost
        self.h = 0  # heuristic cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):  # For heapq
        return self.f < other.f

def get_action():
    return [
        [0, 1, 1], [1, 0, 1], [0, -1, 1], [-1, 0, 1],
        [1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)],
        [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)]
    ]

def heuristic(a, b):
    # Euclidean distance
    return math.hypot(b[0] - a[0], b[1] - a[1])

def collision_check(omap, node_pos):
    for ox, oy in zip(omap[0], omap[1]):
        if node_pos[0] == ox and node_pos[1] == oy:
            return True
    return False

def astar(start, goal, map_obstacle):
    start_node = Node(None, start)
    goal_node = Node(None, goal)

    open_list = []
    closed_set = set()
    visited_plot = set()
    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        _, cur_node = heapq.heappop(open_list)

        if cur_node.position == goal_node.position:
            path = []
            while cur_node is not None:
                path.append(cur_node.position)
                cur_node = cur_node.parent
            return path[::-1]

        closed_set.add(cur_node.position)

        for action in get_action():
            new_pos = (cur_node.position[0] + action[0],
                       cur_node.position[1] + action[1])
            if collision_check(map_obstacle, new_pos):
                continue
            if new_pos in closed_set:
                continue

            child = Node(cur_node, new_pos)
            child.g = cur_node.g + action[2]
            child.h = heuristic(new_pos, goal_node.position)
            child.f = child.g + child.h

            # Open list 중복 방지
            in_open = False
            for _, n in open_list:
                if n.position == child.position and n.f <= child.f:
                    in_open = True
                    break
            if not in_open:
                heapq.heappush(open_list, (child.f, child))

        # 시각화
        if show_animation:
            pos = cur_node.position
            if pos not in visited_plot:
                plt.plot(pos[0], pos[1], 'yo', alpha=0.3)
                visited_plot.add(pos)
            if len(visited_plot) % 100 == 0:
                plt.pause(0.001)

    return None

def main():
    start, goal, omap = map()

    if show_animation:
        plt.figure(figsize=(8, 8))
        plt.plot(start[0], start[1], 'bs', markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs', markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k', markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("A* Algorithm", fontsize=20)

    opt_path = astar(start, goal, omap)
    if opt_path is not None:
        print("Optimal path found!")
        opt_path = np.array(opt_path)
        if show_animation:
            plt.plot(opt_path[:, 0], opt_path[:, 1], "m.-")
            plt.show()
    else:
        print("Path not found.")

if __name__ == "__main__":
    main()
