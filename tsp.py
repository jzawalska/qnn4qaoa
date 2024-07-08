import numpy as np


class TSP:
    def __init__(self, cities_coords):
        self.cities_coords = cities_coords
        self.number_of_cities = len(cities_coords)
        self.normalized_distance_matrix = self.normalize_distance_matrix()

    def calculate_distance_between_points(self, point_A, point_B):
        return np.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2)

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.number_of_cities, self.number_of_cities))
        for i in range(self.number_of_cities):
            for j in range(i, self.number_of_cities):
                distance_matrix[i][j] = self.calculate_distance_between_points(self.cities_coords[i],
                                                                               self.cities_coords[j])
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix

    def normalize_distance_matrix(self):
        distance_matrix = self.calculate_distance_matrix()
        return np.divide(distance_matrix, np.max(distance_matrix))
