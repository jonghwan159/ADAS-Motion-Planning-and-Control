import numpy as np
import matplotlib.pyplot as plt
from map_4 import map


class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent


class RRT(object):
    def __init__(self, start, goal, space, obstacle_list, success_dist_thres=1.0):
        self.start_node = Node(start[0], start[1])
        self.goal_node = Node(goal[0], goal[1])
        self.space = space
        self.obstalce_list = obstacle_list
        self.node_list = []

        self.max_iter = 5000
        self.goal_sample_rate = 0.1
        self.min_u = 1.0
        self.max_u = 3.0
        self.success_dist_thres = success_dist_thres
        self.collision_check_step = 0.2
        self.stepsize = 0.5

    def plan(self):
        self.node_list = [self.start_node]
        for i in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.find_nearest_node(self.node_list, rand_node)
            u = self.stepsize * self.get_random_input(self.min_u, self.max_u)
            new_node = self.create_child_node(nearest_node, rand_node, u)
            if self.is_collide(new_node, self.obstalce_list):
                continue
            if self.is_path_collide(nearest_node, new_node, self.obstalce_list, self.collision_check_step):
                continue
            new_node.set_parent(nearest_node)
            self.node_list.append(new_node)
            if self.check_goal(new_node, self.success_dist_thres):
                print(" [-] GOAL REACHED")
                return self.backtrace_path(new_node)
        return None

    @staticmethod
    def is_same_node(node1, node2):
        return node1.x == node2.x and node1.y == node2.y

    def backtrace_path(self, node):
        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        return path[::-1]

    def get_random_node(self):
        if np.random.rand() < self.goal_sample_rate:
            return self.goal_node
        x = np.random.uniform(self.space[0], self.space[1])
        y = np.random.uniform(self.space[2], self.space[3])
        return Node(x, y)

    def check_goal(self, node, success_dist_thres):
        dx = node.x - self.goal_node.x
        dy = node.y - self.goal_node.y
        return np.hypot(dx, dy) <= success_dist_thres

    @staticmethod
    def create_child_node(nearest_node, rand_node, u):
        theta = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
        new_x = nearest_node.x + u * np.cos(theta)
        new_y = nearest_node.y + u * np.sin(theta)
        return Node(new_x, new_y)

    @staticmethod
    def get_random_input(min_u, max_u):
        return np.random.uniform(min_u, max_u)

    @staticmethod
    def find_nearest_node(node_list, rand_node):
        dists = [np.hypot(node.x - rand_node.x, node.y - rand_node.y) for node in node_list]
        min_index = int(np.argmin(dists))
        return node_list[min_index]

    @staticmethod
    def is_collide(node, obstacle_list):
        for ox, oy, r in obstacle_list:
            if np.hypot(node.x - ox, node.y - oy) <= r:
                return True
        return False

    @staticmethod
    def is_path_collide(node_from, node_to, obstacle_list, check_step=0.2):
        dx = node_to.x - node_from.x
        dy = node_to.y - node_from.y
        dist = np.hypot(dx, dy)
        steps = int(dist / check_step)
        for i in range(steps):
            x = node_from.x + dx * i / steps
            y = node_from.y + dy * i / steps
            for ox, oy, r in obstacle_list:
                if np.hypot(x - ox, y - oy) <= r:
                    return True
        return False


if __name__ == "__main__":
    start, goal, space, obstacle_list = map()

    success_dist_thres = 1.0
    rrt = RRT(start, goal, space, obstacle_list, success_dist_thres)
    path = rrt.plan()
    if path:
        for node in path:
            print(" [-] x = %.2f, y = %.2f " % (node.x, node.y))

        _t = np.linspace(0, 2 * np.pi, 30)
        for obs in obstacle_list:
            x, y, r = obs
            _x = x + r * np.cos(_t)
            _y = y + r * np.sin(_t)
            plt.plot(_x, _y, 'k-')

        goal_x = goal[0] + success_dist_thres * np.cos(_t)
        goal_y = goal[1] + success_dist_thres * np.sin(_t)
        plt.plot(goal_x, goal_y, 'g--')

        for i in range(len(path) - 1):
            node_i = path[i]
            node_ip1 = path[i + 1]
            plt.plot([node_i.x, node_ip1.x], [node_i.y, node_ip1.y], 'r.-')

        plt.axis("equal")
        plt.show()
    else:
        print(" [-] Failed to find path.")