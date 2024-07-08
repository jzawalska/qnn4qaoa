#zaczelam o 16:30
import numpy as np

from reference_method import train_reference
from reference_method_settings import TEST_SIZE, EPOCHS, BATCH_SIZE, TRAINING_SIZE, LEARNING_RATE

# initial_parameters = np.zeros(20)
initial_parameters = np.array([0.05, 1, 0.15, 1, 0.25, 1, 0.35, 1, 0.45, 1, 0.55, 1, 0.65, 1, 0.75, 1, 0.85, 1, 0.95, 1])

name = f"epochs{EPOCHS}_" + \
       f"batch{BATCH_SIZE}_" + \
       f"training_size{TRAINING_SIZE}_" + \
       f"lr{LEARNING_RATE}_" + \
       np.array2string(initial_parameters, separator=',')[:30]

for p in range(1, 10):
    circuit_tensor_test = np.load(f'data/p_{p}/test/circuit_tensor.npy', allow_pickle=True)
    ops_tensor_test = np.load(f'data/p_{p}/test/ops_tensor.npy', allow_pickle=True)

    all_exp_values = []
    all_percent_correct = []

    for i in range(TEST_SIZE):
        print(p, i)
        exp_value, percent_correct, parameter_values = train_reference(p, circuit_tensor_test[i:i + 1], ops_tensor_test[i:i + 1],
                                                           initial_parameters[:(2 * p)])

        all_exp_values.append(exp_value)
        all_percent_correct.append(percent_correct)

        with open(f'./reference_method_results/p_{p}/{name}.csv', 'a') as fd:
            fd.write(
                f"{i},{exp_value[0][0]},{percent_correct},{np.array2string(parameter_values, separator=',', max_line_width=np.inf)}\n")

    mean_exp = np.mean(all_exp_values)
    std_exp = np.std(all_exp_values)
    mean_percent_correct = np.mean(all_percent_correct)
    std_percent_correct = np.std(all_percent_correct)
    with open(f'./reference_method_results/p_{p}/{name}.csv', 'a') as fd:
        text = f"Mean_expectation \n{mean_exp}\nStd_expectation\n{std_exp}\nMean_percent_correct\n{mean_percent_correct}\nStd_percent_correct\n{std_percent_correct}"
        fd.write(text)
