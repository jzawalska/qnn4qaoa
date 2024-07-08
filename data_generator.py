import numpy as np

from qaoa import QAOA_TSP
from tsp import TSP

import json

import tensorflow_quantum as tfq


def generate_data(p, samples_range, type, cities):

    with open('city_coordinates.json') as f:
        data = json.load(f)

    circuits = []
    cost_ops = []
    for i in range(samples_range[0], samples_range[1]):
        print(i)
        tsp_instance = TSP(data['cities_number'][cities][str(i)])
        qaoa_tsp = QAOA_TSP(tsp_instance, p, A_1=4, A_2=4, B=1)
        circuits.append(qaoa_tsp.circuit)
        cost_ops.append([qaoa_tsp.cost_operator])


    circuit_tensor = tfq.convert_to_tensor(circuits)
    cost_tensor = tfq.convert_to_tensor(cost_ops)
    np.save(f'data/p_{p}/{type}/circuit_tensor.npy', circuit_tensor)
    np.save(f'data/p_{p}/{type}/ops_tensor.npy', cost_tensor)
