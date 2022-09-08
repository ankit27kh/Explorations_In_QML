import time
import numpy as np
import qiskit.algorithms.optimizers
from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, transpile, execute
from qiskit.test.mock import FakeLagos
from datasets import sine_wave
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = 42
np.random.seed(seed)
qiskit.utils.algorithm_globals.random_seed = seed

num_points = 25
num_layers = 5
num_qubits = 1
epochs = 10
lr = 0.1
use_noise = True
shots = 1000

device_backend = FakeLagos()

if use_noise:
    print("NOISE ON")
    backend = AerSimulator.from_backend(device_backend)
else:
    print("NOISE OFF")
    backend = AerSimulator()

X_train, X_test, y_train, y_test, dimension = sine_wave(points=num_points)
X_train, X_test, y_train, y_test = (
    X_train.numpy(),
    X_test.numpy(),
    y_train.numpy(),
    y_test.numpy(),
)

num_rots = int(np.ceil(dimension / 3))

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

scaler_x_mm = MinMaxScaler([-1, 1])
X_train_scaled = scaler_x_mm.fit_transform(X_train_scaled)
X_test_scaled = scaler_x_mm.transform(X_test_scaled)

scaler_y_mm = MinMaxScaler([-1, 1])
y_train_scaled = scaler_y_mm.fit_transform(y_train_scaled)
y_test_scaled = scaler_y_mm.transform(y_test_scaled)


def classifier(x, params):
    extra = 3 * num_rots - dimension
    x = np.concatenate([x, np.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = params[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = params[num_thetas:].reshape([num_layers, num_rots, 3])

    qc = QuantumCircuit(num_qubits, 1)

    for k in range(num_layers):
        for i in range(num_rots):
            qc.u(*(x[3 * i: 3 * (i + 1)] * weights[k][i] + varias[k][i]), 0)

    qc.measure(0, 0)

    return qc


parameters = np.random.uniform(-np.pi / 2, np.pi / 2, num_layers * num_rots * 3 * 2)


def circuit(x, params):
    qc = classifier(x, params)
    qc = transpile(qc, backend, optimization_level=0, seed_transpiler=seed)
    res = (
        execute(
            qc,
            backend,
            shots=shots,
            seed_simulator=seed,
            seed_transpiler=seed,
            optimization_level=0,
        )
            .result()
            .get_counts()
    )
    ep = (res.get("1", 0) - res.get("0", 0)) / shots
    return ep


def cost(params, x, y):
    p = circuit(x, params)
    return (y - p) ** 2


def all_cost(params, x, y):
    error = 0
    for xi, yi in zip(x, y):
        error = error + cost(params, xi, yi)
    error = np.sqrt(error / len(x))
    return error


def all_predict(x, params):
    y = []
    for xi in x:
        y.append(circuit(xi, params))
    return np.array(y)


def calculate_metrics(params, x, y):
    y = y.ravel()
    y_pred = all_predict(x, params)
    se = (y_pred - y) ** 2
    ae = abs(y_pred - y)
    rmse = np.sqrt(np.mean(se))
    mae = np.mean(ae)
    return rmse, mae


print("Scores before training:")
rmse_mae_training = calculate_metrics(parameters, X_train_scaled, y_train_scaled)
rmse_mae_testing = calculate_metrics(parameters, X_test_scaled, y_test_scaled)
print(
    "RMSE:",
    round(rmse_mae_training[0], 3),
    "MAE:",
    round(rmse_mae_training[1], 3),
    "on train data before training",
)

print(
    "RMSE:",
    round(rmse_mae_testing[0], 3),
    "MAE:",
    round(rmse_mae_testing[1], 3),
    "on test data before training",
)

opt = qiskit.algorithms.optimizers.SPSA(
    maxiter=epochs,
    learning_rate=lr,
    perturbation=lr * 2,
    # second_order=True,
    # blocking=True,
)
t1 = time.time()

res = opt.optimize(
    parameters.shape[0],
    lambda params: all_cost(params, X_train_scaled, y_train_scaled),
    initial_point=parameters,
)

print("Optimization done", time.time() - t1)

trained_params = res[0]

print("Scores after training:")
rmse_mae_training = calculate_metrics(trained_params, X_train_scaled, y_train_scaled)
rmse_mae_testing = calculate_metrics(trained_params, X_test_scaled, y_test_scaled)
print(
    "RMSE:",
    round(rmse_mae_training[0], 3),
    "MAE:",
    round(rmse_mae_training[1], 3),
    "on train data before training",
)

print(
    "RMSE:",
    round(rmse_mae_testing[0], 3),
    "MAE:",
    round(rmse_mae_testing[1], 3),
    "on test data before training",
)