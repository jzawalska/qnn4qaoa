import numpy as np

from qnn_generalization import train_generalization

from qnn_generalization_settings import TEST_SIZE, EPOCHS, BATCH_SIZE, TRAINING_SIZE, LEARNING_RATE


all_parameters = [
    np.zeros(20),
    np.array(
        [0.05, 1.0, 0.15, 1.0, 0.25, 1.0, 0.35, 1.0, 0.45, 1.0, 0.55, 1.0, 0.65, 1.0, 0.75, 1.0, 0.85, 1.0, 0.95, 1.0]),
]
for id, parameters in enumerate(all_parameters):
    for p in range(1, 10):
        print(f"id {id}, p {p}")
        initial_parameters = parameters[:(2 * p)]
        circuit_tensor = np.load(f'data/p_{p}/training/circuit_tensor.npy', allow_pickle=True)
        ops_tensor = np.load(f'data/p_{p}/training/ops_tensor.npy', allow_pickle=True)
        circuit_tensor_test = np.load(f'data/p_{p}/test/circuit_tensor.npy', allow_pickle=True)
        ops_tensor_test = np.load(f'data/p_{p}/test/ops_tensor.npy', allow_pickle=True)

        name = f"epochs{EPOCHS}_" + \
               f"batch{BATCH_SIZE}_" + \
               f"training_size{TRAINING_SIZE}_" + \
               f"lr{LEARNING_RATE}_" + \
               np.array2string(initial_parameters, separator=',')[:30]

        train_generalization(p, circuit_tensor, ops_tensor, circuit_tensor_test, ops_tensor_test, initial_parameters, name,
            is_verbose=True)
