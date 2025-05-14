import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from map_5 import map

class RRTStar(object):
    def __init__(self, start, goal, config):
        self.G = nx.DiGraph()
        self.G.add_nodes_from([(-1, {'cost': 0, 'x': start[0], 'y': start[1]})])
        self.start = start
        self.goal = goal
        self.config = config

    def sample_free(self, obstacles, space):
        min_x, max_x, min_y, max_y = space
        if np.random.rand() < self.config["goal_sample_rate"]:
            return np.array([self.goal[0], self.goal[1]])
        rand_x = np.random.uniform(min_x, max_x)
        rand_y = np.random.uniform(min_y, max_y)
        return np.array([rand_x, rand_y])

    def get_nearest(self, rand_node):
        min_dist = float('inf')
        nearest_id = None
        for node_id in self.G.nodes:
            node = self.get_node(node_id)
            dist = np.linalg.norm(rand_node - node)
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        return nearest_id

    def steer(self, node_from, node_to, u=None):
        if u is None:
            u = self.config["eta"]
        direction = node_to - node_from
        distance = np.linalg.norm(direction)
        if distance <= u:
            return node_to
        direction = direction / distance
        return node_from + u * direction

    def get_node(self, node_id):
        node = np.array([self.G.nodes[node_id]['x'], self.G.nodes[node_id]['y']])
        return node

    def is_collision_free(self, node_from, node_to, obstacles, step=0.2):
        direction = node_to - node_from
        distance = np.linalg.norm(direction)
        steps = int(distance / step)
        for i in range(steps + 1):
            x = node_from[0] + direction[0] * i / steps
            y = node_from[1] + direction[1] * i / steps
            for obs in obstacles:
                if obs.is_inside(x, y):
                    return False
        return True

    def get_near_node_ids(self, new_node, draw):
        n = len(self.G.nodes)
        r = self.config["gamma_rrt_star"] * np.sqrt(np.log(n) / n)
        near_ids = []
        for node_id in self.G.nodes:
            node = self.get_node(node_id)
            if np.linalg.norm(new_node - node) <= r:
                near_ids.append(node_id)
        return near_ids

    def add_node(self, node_id, x, y):
        self.G.add_node(node_id, x=x, y=y)

    def get_node_cost(self, node_id):
        return self.G.nodes[node_id]['cost']

    def get_distance(self, node_from_id, node_to_id):
        node_from = self.get_node(node_from_id)
        node_to = self.get_node(node_to_id)
        return np.linalg.norm(node_to - node_from)

    def add_edge(self, node_from_id, node_to_id):
        self.G.add_edge(node_from_id, node_to_id)

    def set_node_cost(self, node_id, cost):
        self.G.nodes[node_id]['cost'] = cost

    def get_parent(self, node_id):
        parents = list(self.G.predecessors(node_id))
        if len(parents) > 0:
            return parents[0]
        else:
            return None

    def remove_edge(self, node_from_id, node_to_id):
        if self.G.has_edge(node_from_id, node_to_id):
            self.G.remove_edge(node_from_id, node_to_id)

    def check_goal_by_id(self, node_id):
        node = self.G.nodes[node_id]
        dx = node['x'] - self.goal[0]
        dy = node['y'] - self.goal[1]
        dist = np.hypot(dx, dy)
        return dist < self.config["goal_range"]


if __name__ == '__main__':
    start, goal, space, obstacles = map()
    for obs in obstacles:
        obs.plot()

    config = {
        "eta": 3.0,
        "gamma_rrt_star": 4.0,
        "goal_sample_rate": 0.05,
        "min_u": 1.0,
        "max_u": 3.0,
        "goal_range": 1.0
    }

    rrt_star = RRTStar(start, goal, config)

    is_first_node = True
    goal_node_id = None

    for i in range(1000):
        rand_node = rrt_star.sample_free(obstacles, space)
        nearest_node_id = rrt_star.get_nearest(rand_node)
        nearest_node = rrt_star.get_node(nearest_node_id)
        new_node = rrt_star.steer(nearest_node, rand_node)

        if rrt_star.is_collision_free(nearest_node, new_node, obstacles):
            near_node_ids = rrt_star.get_near_node_ids(new_node, draw=True)
            rrt_star.add_node(i, new_node[0], new_node[1])

            if is_first_node:
                rrt_star.add_edge(-1, i)
                is_first_node = False

            min_node_id = nearest_node_id
            min_cost = rrt_star.get_node_cost(nearest_node_id) + rrt_star.get_distance(nearest_node_id, i)

            for near_node_id in near_node_ids:
                near_node = rrt_star.get_node(near_node_id)
                if rrt_star.is_collision_free(near_node, new_node, obstacles):
                    cost = rrt_star.get_node_cost(near_node_id) + rrt_star.get_distance(near_node_id, i)
                    if cost < min_cost:
                        min_node_id = near_node_id
                        min_cost = cost

            rrt_star.set_node_cost(i, min_cost)
            rrt_star.add_edge(min_node_id, i)

            for near_node_id in near_node_ids:
                near_node = rrt_star.get_node(near_node_id)
                if rrt_star.is_collision_free(new_node, near_node, obstacles):
                    cost = rrt_star.get_node_cost(i) + rrt_star.get_distance(i, near_node_id)
                    if cost < rrt_star.get_node_cost(near_node_id):
                        parent_node_id = rrt_star.get_parent(near_node_id)
                        if parent_node_id is not None:
                            rrt_star.remove_edge(parent_node_id, near_node_id)
                            rrt_star.add_edge(i, near_node_id)

            if rrt_star.check_goal_by_id(i):
                goal_node_id = i
                break

    for e in rrt_star.G.edges:
        v_from = rrt_star.G.nodes[e[0]]
        v_to = rrt_star.G.nodes[e[1]]
        plt.plot([v_from['x'], v_to['x']], [v_from['y'], v_to['y']], 'b-')
        #plt.text(v_to['x'], v_to['y'], str(e[1]), fontsize=6)

    if goal_node_id is not None:
        path = nx.shortest_path(rrt_star.G, source=-1, target=goal_node_id)
        xs, ys = [], []
        for node_id in path:
            node = rrt_star.G.nodes[node_id]
            xs.append(node['x'])
            ys.append(node['y'])
        plt.plot(xs, ys, 'r-', lw=3)

    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bx')
    plt.axis("equal")
    plt.show()
