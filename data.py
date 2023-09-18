import numpy as np
import torch
from torch.utils.data import Dataset

class LineDataset(Dataset):
    def __init__(self, num_input_data: int = 10, num_samples_per_line: int = 10):
        """
        Initialize the LineDataset.

        Parameters:
        - num_input_data: Number of lines to create in the synthetic data
        - num_samples_per_line: Number of samples from a line to feed into the model
        """
        # Generate random slopes for 5 lines
        np.random.seed(0)
        line_params = np.random.uniform(0, 1.0, (num_input_data, 2))
        self.x, self.lines = self.generate_lines(line_params, num_points=num_samples_per_line)
        self.sampled_points = self.sample_points_from_lines(self.x, self.lines, num_samples_per_line=num_samples_per_line)
        
        self.num_samples_per_line = num_samples_per_line

    def __len__(self):
        return len(self.lines)
    
    def sample_points_from_lines(self, x: np.array, lines: list, num_samples_per_line: int = 10):
        """
        Sample points from given lines.

        Parameters:
        - x: x input values.
        - lines: list of tuples (slope, bias, y).
        - num_samples_per_line: Number of points to sample per line.

        Returns:
        - sampled_points: List of tuples, each tuple being (x, y) of a sampled point.
        """
        sampled_points = []
        for y_values in lines:
            indices = np.random.choice(len(x), num_samples_per_line, replace=False)
            for index in indices:
                sampled_points.append((x[index], y_values[2][index]))
        return sampled_points

    def generate_lines(self, line_params: np.array, x_range: tuple = (-1, 1), num_points: int = 400):
        """
        Generate y values for given slopes and x values.

        Parameters:
        - line_params: Array of parameters for a line, 0th column is the slope and 1st column is the bias.
        - x_range: Tuple indicating the range of x values.
        - num_points: Number of x values to generate.

        Returns:
        - x: x values.
        - lines: list of tuples (slope, bias, y).
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        # [m, mx + b]
        lines = [(data[0], data[1], data[0] * x + data[1]) for i, data in enumerate(line_params)]
        return x, lines

    def __getitem__(self, idx):
        # y = mx + b
        slope, bias, y = self.lines[idx]
        indices = np.random.choice(len(y), self.num_samples_per_line, replace=False)
        y = y[indices]
        x = self.x[indices]
        input_tensor = np.concatenate([x,y],axis=0)
        target = np.stack([slope, bias])

        # torch.atan(torch.tensor(slope))
        return torch.tensor(input_tensor), torch.tensor(target)

