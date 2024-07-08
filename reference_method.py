import tensorflow as tf
import tensorflow_quantum as tfq

import numpy as np
import sympy

from qnn_generalization_settings import LEARNING_RATE
from reference_method_settings import EPOCHS


def generate_sympy_parameters(p):
    return sympy.symbols('parameter_:%d' % (2 * p))


class QFFNN(tf.keras.layers.Layer):
    def __init__(self, symbol_names, p, initial_parameters):
        super(QFFNN, self).__init__()
        self.symbol_names = symbol_names
        self.hidden = tf.keras.layers.Dense(2 * p, name="hidden", use_bias=True,
                                            bias_initializer=tf.keras.initializers.Constant(initial_parameters))
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


def train_reference(p, circuit_tensor_test, ops_tensor_test, initial_parameters):
    symbols = generate_sympy_parameters(p)

    parametrized_circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    operator_input = tf.keras.Input(shape=(1,), dtype=tf.dtypes.string)
    parameters_input = tf.keras.Input(shape=(1,))

    qffnn = QFFNN(symbols, p, initial_parameters)
    output = qffnn([parametrized_circuit_input, operator_input, parameters_input])

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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.mean_absolute_error,
        loss_weights=[0, 1]
    )

    initial_params_test = np.zeros((1, 1)).astype(np.float32)
    history = model.fit(
        x=[
            circuit_tensor_test,
            ops_tensor_test,
            initial_params_test,
        ],
        y=[
            np.zeros((1, p * 2)),
            np.zeros((1, 1)),  # the closer to 0 the better the result
        ],
        epochs=EPOCHS,
        verbose=0)

    parameter_values, exp_val = model.predict([circuit_tensor_test, ops_tensor_test, initial_params_test])

    samples_amount = 2 ** 16
    sample_layer = tfq.layers.Sample()
    output = sample_layer(circuit_tensor_test,
                          symbol_names=symbols,
                          symbol_values=parameter_values,
                          repetitions=samples_amount)

    from collections import Counter

    results = output.numpy()[0].astype(str).tolist()
    results_to_display = [''.join(result) for result in results]
    correct_results = (
        "0001100001000010", "0010010010000001", "0100100000010010", "1000000100100100", "1000010000100001",
        "0100001000011000",
        "0001001001001000", "0010000110000100", "0100000110000010", "0010100000010100", "0001010000101000",
        "0001100000100100",
        "1000000101000010", "1000001001000001", "0100001010000001", "0100000100101000", "0010010000011000",
        "0100100000100001",
        "1000001000010100", "0001001010000100", "0001010010000010", "0010000101001000", "1000010000010010",
        "0010100001000001")
    counts = Counter(results_to_display)

    correct_results_count = sum(counts[result] for result in correct_results)
    percent_correct_results = round(correct_results_count / samples_amount * 100, 2)

    return exp_val, percent_correct_results, parameter_values
