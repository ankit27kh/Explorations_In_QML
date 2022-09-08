from jax.config import config

config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import optax
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import jax
from sklearn.svm import SVR
from circuit_layers import (
    get_parameters,
    layer_1,
    layer_2,
    layer_3,
    layer_4,
    layer_5,
    layer_6,
    layer_7,
    layer_8,
)
from datasets import (
    polynomial,
    sine_wave,
    exponential,
    mod_x,
    boston_housing,
    UCI_air_quality,
    UCI_auto_mpg,
    UCI_automobile,
    UCI_bike_share,
    UCI_computer_hardware,
    UCI_energy_efficiency,
    UCI_forest_fires,
    UCI_student_performance,
    UCI_wine_quality,
)
import pennylane as qml
import jax.numpy as jnp

X_train, X_test, y_train, y_test, dimension = polynomial()

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

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = (
    jnp.asarray(X_train_scaled),
    jnp.asarray(X_test_scaled),
    jnp.asarray(y_train_scaled),
    jnp.asarray(y_test_scaled),
)

num_shots = None
num_layers = 4
num_qubits = 2

thetas, num_qubits = get_parameters(
    layer=2,
    dimension=dimension,
    num_layers=num_layers,
    num_qubits=num_qubits,
)
learning_rate = 0.01
epochs = 500
batch_size = int(jnp.ceil(len(X_train) * 0.1))


def iterate_batches(X, Y, batch_size):
    batch_list_x = []
    batch_x = []
    for x in X:
        if len(batch_x) < batch_size:
            batch_x.append(x)

        else:
            batch_list_x.append(jnp.asarray(batch_x))
            batch_x = []
    if len(batch_x) != 0:
        batch_list_x.append(jnp.asarray(batch_x))

    batch_list_y = []
    batch_y = []
    for y in Y:
        if len(batch_y) < batch_size:
            batch_y.append(y)

        else:
            batch_list_y.append(jnp.asarray(batch_y))
            batch_y = []
    if len(batch_y) != 0:
        batch_list_y.append(jnp.asarray(batch_y))
    return batch_list_x, batch_list_y


original_parameters = thetas.copy()

dev = qml.device("default.qubit.jax", wires=num_qubits, shots=num_shots)

projector = jnp.zeros((2 ** num_qubits, 2 ** num_qubits))
projector = projector.at[0, 0].set(1)


@jax.jit
@qml.qnode(dev, interface="jax")
def kernel(parameters, x1, x2):
    layer(parameters, x1)
    qml.Barrier(wires=range(num_qubits))
    qml.adjoint(layer)(parameters, x2)
    return qml.expval(qml.Hermitian(projector.tolist(), wires=range(num_qubits)))


def layer(parameters, x):
    layer_2(parameters, x, num_layers, num_qubits)


def individual_kernel(parameters, x, y):
    return kernel(parameters, x, y)


mapped_kernel = jax.vmap(
    jax.vmap(individual_kernel, in_axes=(None, None, 0)), in_axes=(None, 0, None)
)


def mapped_kernel_params(params):
    return lambda X, Y: mapped_kernel(params, X, Y)


def metrics(x, y):
    y_pred_ = svm.predict(x)
    rmse = jnp.sqrt(mean_squared_error(y_pred_.reshape(-1, 1), y))
    mae = mean_absolute_error(y_pred_.reshape(-1, 1), y)
    return rmse, mae


def target_alignment(X, Y, parameters):
    Kx = mapped_kernel_params(parameters)(X, X)
    _Y = jnp.array(Y)
    Ky = jnp.outer(_Y, _Y)

    N = len(X)

    def kernel_center(ker):
        sq = jnp.ones([N, N])
        c = jnp.identity(N) - sq / N
        return jnp.matmul(c, jnp.matmul(ker, c))

    KxC = kernel_center(Kx)
    KyC = kernel_center(Ky)

    kxky = jnp.trace(jnp.matmul(KxC.T, KyC))
    kxkx = jnp.linalg.norm(KxC)
    kyky = jnp.linalg.norm(KyC)

    kta = kxky / kxkx / kyky
    return kta


@jax.jit
def cost(params, x, y):
    return -target_alignment(x, y, params)


training_kernel_matrix = mapped_kernel(thetas, X_train_scaled, X_train_scaled)
testing_kernel_matrix = mapped_kernel(thetas, X_test_scaled, X_train_scaled)

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

kta_values = [-cost(thetas, X_train_scaled, y_train_scaled)]


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        kta_i, grads = jax.value_and_grad(cost)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, kta_i

    for i in range(epochs):
        x_batches, y_batches = iterate_batches(
            X_train_scaled, y_train_scaled, batch_size
        )
        epoch_cost = []
        for x_batch, y_batch in zip(x_batches, y_batches):
            params, opt_state, kta_i = step(params, opt_state, x_batch, y_batch)
            epoch_cost.append(-kta_i)
        kta_values.append(jnp.mean(jnp.asarray(epoch_cost)))
    return params


optimizer = optax.adam(learning_rate)
thetas = fit(thetas, optimizer)

trained_parameters = thetas.copy()

training_kernel_matrix = mapped_kernel(thetas, X_train_scaled, X_train_scaled)
testing_kernel_matrix = mapped_kernel(thetas, X_test_scaled, X_train_scaled)

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

plt.plot(kta_values)
plt.show()

dev_ = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev_, expansion_strategy="device")
def kernel(parameters, x1, x2):
    layer(parameters, x1)
    qml.Barrier(wires=range(num_qubits))
    qml.adjoint(layer)(parameters, x2)
    return qml.expval(qml.Hermitian(projector.tolist(), wires=range(num_qubits)))


fig, ax = qml.draw_mpl(kernel, expansion_strategy="device")(
    thetas, X_train_scaled[0], X_train_scaled[0]
)
fig.show()

resources = qml.specs(kernel)(thetas, X_train_scaled[0], X_train_scaled[0])
print(resources)

gate_types = dict(resources["gate_sizes"])
total_gates = resources["num_operations"]
depth = resources["depth"]
num_params = thetas.ravel().shape[0]

"""
import pickle

name = 'UCI_automobile_type_1_kernel'
save_data = []
save_data.extend(
    [num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters.numpy(),
     trained_parameters, kta_values, before_training_metrics, before_testing_metrics, after_training_metrics,
     after_testing_metrics])
with open(f'new_data_/Regression/Kernel/{name}_{num_qubits}_{num_layers}.pkl', 'wb') as file:
    pickle.dump(save_data, file)
"""
