import time
import matplotlib.pyplot as plt
import numpy as np
import qiskit.algorithms.optimizers
from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, transpile, execute
from qiskit.test.mock import FakeLagos
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from datasets import polynomial
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

X_train, X_test, y_train, y_test, dimension = polynomial(points=num_points)
X_train, X_test, y_train, y_test = X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy()


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

parameters = np.random.uniform(-np.pi / 2, np.pi / 2, num_layers * num_rots * 3 * 2)

plt.plot(X_test_scaled, y_test_scaled, ":.", label="og")


def feature_map(x, params):
    extra = 3 * num_rots - dimension
    x = np.concatenate([x, np.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = params[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = params[num_thetas:].reshape([num_layers, num_rots, 3])

    qc = QuantumCircuit(num_qubits, 1)

    for k in range(num_layers):
        for i in range(num_rots):
            qc.u(*(x[3 * i: 3 * (i + 1)] * weights[k][i] + varias[k][i]), 0)

    return qc


def adjoint_feature_map(x, params):
    extra = 3 * num_rots - dimension
    x = np.concatenate([x, np.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = params[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = params[num_thetas:].reshape([num_layers, num_rots, 3])

    qc = QuantumCircuit(num_qubits, 1)

    angles_list = []
    for k in range(num_layers):
        for i in range(num_rots):
            angles_list.append((x[3 * i: 3 * (i + 1)] * weights[k][i] + varias[k][i]))

    for i in range(len(angles_list)):
        angles = angles_list[-(i + 1)]
        qc.u(-angles[0], -angles[2], -angles[1], 0)

    return qc


def kernel(x1, x2, params):
    qc_1 = feature_map(x1, params)
    qc_2 = adjoint_feature_map(x2, params)
    qc = qc_1.compose(qc_2)
    qc.measure(0, 0)

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

    k = res.get("0", 0) / shots

    return k


def kernel_matrix(x1, x2, params):
    m = len(x1)
    n = len(x2)
    matrix = np.zeros([m, n])
    for i, p1 in enumerate(x1):
        for j, p2 in enumerate(x2):
            matrix[i][j] = kernel(p1, p2, params)

    return matrix


def metrics(x, y):
    y_pred_ = svm.predict(x)
    rmse = np.sqrt(mean_squared_error(y_pred_.reshape(-1, 1), y))
    mae = mean_absolute_error(y_pred_.reshape(-1, 1), y)
    return rmse, mae


training_kernel_matrix = kernel_matrix(X_train_scaled, X_train_scaled, parameters)
testing_kernel_matrix = kernel_matrix(X_test_scaled, X_train_scaled, parameters)

training_kernel_matrix = (training_kernel_matrix + training_kernel_matrix.T) / 2

svm = SVR(kernel="precomputed")
svm.fit(training_kernel_matrix, y_train_scaled.ravel())
before_training_metrics = metrics(training_kernel_matrix, y_train_scaled)
before_testing_metrics = metrics(testing_kernel_matrix, y_test_scaled)
print(
    round(before_training_metrics[0], 3),
    round(before_training_metrics[1], 3),
    "before training train data",
)
print(
    round(before_testing_metrics[0], 3),
    round(before_testing_metrics[1], 3),
    "before training test data",
)

plt.plot(X_test_scaled, svm.predict(testing_kernel_matrix), ":.", label="before")


def target_alignment(X, Y, params):
    Kx = kernel_matrix(X, X, params)
    _Y = np.array(Y)
    Ky = np.outer(_Y, _Y)

    N = len(X)

    def kernel_center(ker):
        sq = np.ones([N, N])
        c = np.identity(N) - sq / N
        return np.matmul(c, np.matmul(ker, c))

    KxC = kernel_center(Kx)
    KyC = kernel_center(Ky)

    kxky = np.trace(np.matmul(KxC.T, KyC))
    kxkx = np.linalg.norm(KxC)
    kyky = np.linalg.norm(KyC)

    kta = kxky / kxkx / kyky
    return kta


def cost(params, x, y):
    return -target_alignment(x, y, params)


print(-cost(parameters, X_train_scaled, y_train_scaled))

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
    lambda params: cost(params, X_train_scaled, y_train_scaled),
    initial_point=parameters,
)

print("Optimization done", time.time() - t1)

trained_params = res[0]

training_kernel_matrix = kernel_matrix(X_train_scaled, X_train_scaled, trained_params)
testing_kernel_matrix = kernel_matrix(X_test_scaled, X_train_scaled, trained_params)
training_kernel_matrix = (training_kernel_matrix + training_kernel_matrix.T) / 2

svm = SVR(kernel="precomputed")
svm.fit(training_kernel_matrix, y_train_scaled.ravel())
after_training_metrics = metrics(training_kernel_matrix, y_train_scaled)
after_testing_metrics = metrics(testing_kernel_matrix, y_test_scaled)
print(
    round(after_training_metrics[0], 3),
    round(after_training_metrics[1], 3),
    "after training train data",
)
print(
    round(after_testing_metrics[0], 3),
    round(after_testing_metrics[1], 3),
    "after training test data",
)

print(-cost(trained_params, X_train_scaled, y_train_scaled))

plt.plot(X_test_scaled, svm.predict(testing_kernel_matrix), ":.", label="after")
plt.legend()
plt.show()
