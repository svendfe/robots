#!/usr/bin/env python3
"""
Interactive RRT* Path Planner
Improvements:
1. Click-to-select Start/Goal
2. Live visualization of tree growth
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Optional
import csv

class Node:
    """Node in the RRT tree"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional[Node] = None
        self.cost = 0.0

class RRTStar:
    """RRT* path planning algorithm with live plotting"""
    
    def __init__(self, grid_map: np.ndarray, step_size: float = 1.5, 
                 search_radius: float = 3.0, max_iterations: int = 1000):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iterations = max_iterations
        self.nodes: List[Node] = []
        
    def distance(self, n1: Node, n2: Node) -> float:
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    
    def is_collision_free(self, n1: Node, n2: Node) -> bool:
        steps = int(np.ceil(self.distance(n1, n2) * 2))
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(np.round(n1.x + t * (n2.x - n1.x)))
            y = int(np.round(n1.y + t * (n2.y - n1.y)))
            if x < 0 or x >= self.width or y < 0 or y >= self.height: return False
            if self.grid_map[y, x] >= 0.5: return False
        return True
    
    def get_path_cost(self, n1: Node, n2: Node) -> float:
        cost = self.distance(n1, n2)
        steps = int(np.ceil(self.distance(n1, n2) * 2))
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(np.round(n1.x + t * (n2.x - n1.x)))
            y = int(np.round(n1.y + t * (n2.y - n1.y)))
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.grid_map[y, x] == 0.5: cost += 1.5
        return cost
    
    def find_nearest_node(self, random_node: Node) -> Node:
        nearest = self.nodes[0]
        min_dist = self.distance(nearest, random_node)
        for node in self.nodes:
            dist = self.distance(node, random_node)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest
    
    def steer(self, from_node: Node, to_node: Node) -> Node:
        dist = self.distance(from_node, to_node)
        if dist < self.step_size:
            return Node(to_node.x, to_node.y)
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        return Node(from_node.x + self.step_size * np.cos(theta),
                   from_node.y + self.step_size * np.sin(theta))
    
    def find_near_nodes(self, new_node: Node) -> List[Node]:
        return [node for node in self.nodes 
                if self.distance(node, new_node) < self.search_radius]
    
    def draw_live(self, ax, start, goal, iteration):
        """Helper to draw the tree during execution"""
        ax.clear()
        
        # Draw Map
        cmap_colors = ['#f3f4f6', '#fef3c7', '#1f2937'] 
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap_colors)
        ax.imshow(self.grid_map, cmap=cmap, origin='lower', alpha=0.8, vmin=0, vmax=1)
        
        # Draw Tree edges
        for node in self.nodes:
            if node.parent:
                ax.plot([node.parent.x, node.x], [node.parent.y, node.y], 'c-', alpha=0.5, linewidth=0.5)
        
        # Draw Start/Goal
        ax.plot(start[0], start[1], 'bo', markersize=8, label='Start')
        ax.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
        
        ax.set_title(f"RRT* Iteration: {iteration}/{self.max_iterations}")
        plt.pause(0.01)

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int], ax=None) -> Tuple[List[Node], List[Tuple[float, float]], bool]:
        start_node = Node(float(start[0]), float(start[1]))
        goal_node = Node(float(goal[0]), float(goal[1]))
        self.nodes = [start_node]
        goal_reached = False
        
        print(f"Starting planning...")
        
        for iteration in range(self.max_iterations):
            # ----------------------
            # Standard RRT* Logic
            # ----------------------
            if np.random.random() < 0.1:
                random_node = goal_node
            else:
                random_node = Node(np.random.random() * self.width, np.random.random() * self.height)
            
            nearest = self.find_nearest_node(random_node)
            new_node = self.steer(nearest, random_node)
            
            if not self.is_collision_free(nearest, new_node):
                continue
            
            near_nodes = self.find_near_nodes(new_node)
            min_cost = nearest.cost + self.get_path_cost(nearest, new_node)
            best_parent = nearest
            
            for near_node in near_nodes:
                if self.is_collision_free(near_node, new_node):
                    cost = near_node.cost + self.get_path_cost(near_node, new_node)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near_node
            
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewiring
            for near_node in near_nodes:
                if near_node == best_parent: continue
                new_cost = new_node.cost + self.get_path_cost(new_node, near_node)
                if new_cost < near_node.cost and self.is_collision_free(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = new_cost
            
            # Goal Check
            if not goal_reached and self.distance(new_node, goal_node) < self.step_size:
                if self.is_collision_free(new_node, goal_node):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + self.get_path_cost(new_node, goal_node)
                    self.nodes.append(goal_node)
                    goal_reached = True
                    print(f"Goal reached at iteration {iteration}!")

            # ----------------------
            # Visualization Update
            # ----------------------
            if ax and iteration % 100 == 0:  # Update every 100 iterations
                self.draw_live(ax, start, goal, iteration)

        # Final Path Extraction
        path = []
        if goal_reached:
            current = goal_node
            while current is not None:
                path.insert(0, (current.x, current.y))
                current = current.parent
            print(f"Final Path Cost: {goal_node.cost:.2f}")
        
        return self.nodes, path, goal_reached

def select_points(grid_map):
    """Interactive Start/Goal selection"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show map
    cmap_colors = ['#f3f4f6', '#fef3c7', '#1f2937'] 
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_colors)
    ax.imshow(grid_map, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax.set_title("Click START point, then click GOAL point")
    
    print("Please click the Start point on the map window...")
    pts = plt.ginput(2, timeout=-1) # Wait for 2 clicks
    plt.close()
    
    if len(pts) < 2:
        raise ValueError("Points were not selected correctly.")
        
    start = (int(pts[0][0]), int(pts[0][1]))
    goal = (int(pts[1][0]), int(pts[1][1]))
    
    print(f"Selected Start: {start}")
    print(f"Selected Goal: {goal}")
    return start, goal

def load_map(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [[float(val) for val in row] for row in reader]
    return np.array(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('map_file', help='Path to CSV map file')
    # Made start/end optional now
    parser.add_argument('--start', nargs=2, type=int, help='Start (x y)')
    parser.add_argument('--end', nargs=2, type=int, help='Goal (x y)')
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()
    
    grid_map = load_map(args.map_file)
    
    # Interactive Selection Logic
    if args.start is None or args.end is None:
        try:
            start, goal = select_points(grid_map)
        except Exception as e:
            print(f"Error selecting points: {e}")
            return
    else:
        start, goal = tuple(args.start), tuple(args.end)

    # Setup Plot for Live Viz
    plt.ion() # Interactive mode on
    fig, ax = plt.subplots(figsize=(10, 10))
    
    planner = RRTStar(grid_map, max_iterations=args.iterations)
    nodes, path, success = planner.plan(start, goal, ax=ax)
    
    # Final Draw
    planner.draw_live(ax, start, goal, args.iterations)
    if len(path) > 0:
        path_arr = np.array(path)
        ax.plot(path_arr[:,0], path_arr[:,1], 'g-', linewidth=3, label='Final Path')
    
    plt.ioff() # Interactive mode off
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()