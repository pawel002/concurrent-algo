import matplotlib.pyplot as plt
import sys
import os

def load_coordinates(filename):
    coords = {}
    with open(filename, 'r') as f:
        header = f.readline()
        
        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coords[node_id] = (x, y)
            
    return coords

def load_solution(filename):
    path = []
    cost = 0.0
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        cost = float(lines[0].split(':')[1].strip())
        path = list(map(float, lines[1].split()))
            
    return cost, path

def visualize(coords, path, cost):
    x_vals = []
    y_vals = []
    
    for node_id in path:
        if node_id in coords:
            x_vals.append(coords[node_id][0])
            y_vals.append(coords[node_id][1])
        else:
            print(f"Warning: Node {node_id} in solution but not in coords file.")

    plt.figure(figsize=(10, 8))
    plt.plot(x_vals, y_vals, c='blue', linewidth=2, zorder=1, label='Path')
    
    all_x = [c[0] for c in coords.values()]
    all_y = [c[1] for c in coords.values()]
    plt.scatter(all_x, all_y, c='red', s=100, zorder=2, label='Cities')

    for node_id, (x, y) in coords.items():
        plt.text(x, y, str(node_id), fontsize=12, ha='right', va='bottom', weight='bold')

    start_x, start_y = coords[path[0]]
    plt.scatter(start_x, start_y, c='green', s=150, zorder=3, label='Start/End')

    plt.title(f"TSP Solution\nTotal Cost: {cost:.4f}", fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.axis('equal') 
    
    plt.tight_layout()
    plt.savefig("data/benchmarks/vis.png")
    plt.show()

if __name__ == "__main__":
    DIR = "data/"
    coordinates = load_coordinates(DIR + "coords.txt")
    cost, solution_path = load_solution(DIR + "solution.txt")
    visualize(coordinates, solution_path, cost)