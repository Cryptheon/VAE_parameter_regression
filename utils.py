import matplotlib.pyplot as plt

def plot_sampled_points(sampled_points: tuple):
    """
    Plot sampled points.

    Parameters:
    - sampled_points: List of tuples, each tuple being (x, y) of a sampled point.
    """
    plt.figure(figsize=(8, 8))
    for point in sampled_points:
        plt.scatter(*point, marker='o', color='blue', s=2)
    
    plt.title("Sampled points from lines originating from the origin")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()