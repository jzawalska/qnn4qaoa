import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from qnn_generalization_settings import TEST_SIZE, EPOCHS, BATCH_SIZE, TRAINING_SIZE, LEARNING_RATE


def generate_sympy_parameters(p):
    return sympy.symbols('parameter_:%d' % (2 * p))


class QFFNN(tf.keras.layers.Layer):
    def __init__(self, symbol_names, p, initial_parameters):
        super(QFFNN, self).__init__()
        self.symbol_names = symbol_names
        self.hidden = tf.keras.layers.Dense(2 * p, name="hidden", use_bias=True, bias_initializer=tf.keras.initializers.Constant(initial_parameters))
        self.expectation = tfq.layers.Expectation(name="expectation")

    def call(self, inputs):
        parameterized_circuit = inputs[0]  # qaoa_circuits
        cost_operator = inputs[1]  # cost_hams
        params = inputs[2]  # qaoa_parameters

        hidden_out = self.hidden(params)

        expectation_value = self.expectation(parameterized_circuit,
                                             symbol_values=hidden_out,
                                             operators=cost_operator,
                                             symbol_names=self.symbol_names)

        return [hidden_out, expectation_value]


def train_generalization(p, circuit_tensor, ops_tensor, circuit_tensor_test, ops_tensor_test, initial_parameters, name, is_verbose=False):
    parametrized_circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    operator_input = tf.keras.Input(shape=(1,), dtype=tf.dtypes.string)
    parameters_input = tf.keras.Input(shape=(1,))

    symbols = generate_sympy_parameters(p)
    qffnn = QFFNN(symbols, p, initial_parameters)
    output = qffnn([parametrized_circuit_input,
                    operator_input,
                    parameters_input])

    model = tf.keras.Model(
        inputs=[
            parametrized_circuit_input,
            operator_input,
            parameters_input
        ],
        outputs=[
            output[0],  # array of optimized 2p parameters
            output[1],  # expectation value
        ])

    @tf.function
    def loss_function(unused, outputs):
        for output in outputs:
            tf.print(output, summarize=20, end=',', output_stream=f"file://outputs/p_{p}/{name}.csv")
        return tf.keras.losses.mean_absolute_error(0, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss=loss_function,
                  loss_weights=[0, 1])

    initial_params = np.zeros((TRAINING_SIZE, 1)).astype(np.float32)  # zeros
    x_in = [circuit_tensor, ops_tensor, initial_params]
    y_in = [np.zeros((TRAINING_SIZE, 1)), np.zeros((TRAINING_SIZE, 1))]

    history = model.fit(
        x=x_in,
        y=y_in,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=is_verbose)

    # initial_params_test = np.zeros((TEST_SIZE, 1)).astype(np.float32)
    # x_test = [circuit_tensor_test, ops_tensor_test, initial_params_test]
    # y_test = [np.zeros((TEST_SIZE, 1)), np.zeros((TEST_SIZE, 1))]
    # expectation_value_test = model.evaluate(x_test, y_test)
    # print("expectation_value_test", expectation_value_test)
    # parameter_values = model.predict([circuit_tensor_test, ops_tensor_test, initial_params_test])[0]
    # print(f"Parameter values:\n {parameter_values}")

    samples_amount = 2 ** 16
    sample_layer = tfq.layers.Sample()

    with open(f'./results/p_{p}/{name}.csv', 'a') as fd:
        fd.write('test_city_id,exp_value,percent_correct,parameter_values\n')

    all_exp_values = []
    all_percent_correct = []

    for i in range(TEST_SIZE):
        initial_params_test = np.zeros((1, 1)).astype(np.float32)
        parameter_values, exp_value = model.predict([circuit_tensor_test[i:i+1], ops_tensor_test[i:i+1], initial_params_test])
        output = sample_layer(circuit_tensor_test[i:i+1],
                              symbol_names=symbols,
                              symbol_values=parameter_values,
                              repetitions=samples_amount)

        from collections import Counter

        results = output.numpy()[0].astype(str).tolist()
        results_to_display = [''.join(result) for result in results]
        correct_results = ("0001100001000010","0010010010000001","0100100000010010","1000000100100100","1000010000100001","0100001000011000","0001001001001000","0010000110000100","0100000110000010","0010100000010100","0001010000101000","0001100000100100","1000000101000010","1000001001000001","0100001010000001", "0100000100101000", "0010010000011000", "0100100000100001", "1000001000010100", "0001001010000100", "0001010010000010","0010000101001000", "1000010000010010", "0010100001000001")
        counts = Counter(results_to_display)

        correct_results_count = sum(counts[result] for result in correct_results)
        percent_correct = round(correct_results_count / samples_amount * 100, 2)

        all_exp_values.append(exp_value[0][0])
        all_percent_correct.append(percent_correct)

        with open(f'./results/p_{p}/{name}.csv', 'a') as fd:
            fd.write(f"{i},{exp_value[0][0]},{percent_correct},{np.array2string(parameter_values, separator=',', max_line_width=np.inf)}\n")


    model.save_weights(f"./models/weights/p_{p}/{name}")
    np.save(f"./models/history/p_{p}/{name}", history.history)

    mean_exp = np.mean(all_exp_values)
    std_exp = np.std(all_exp_values)
    mean_percent_correct = np.mean(all_percent_correct)
    std_percent_correct = np.std(all_percent_correct)

    with open(f'./results/p_{p}/{name}.csv', 'a') as fd:
        text = f"Mean_expectation \n{mean_exp}\nStd_expectation\n{std_exp}\nMean_percent_correct\n{mean_percent_correct}\nStd_percent_correct\n{std_percent_correct}"
        fd.write(text)