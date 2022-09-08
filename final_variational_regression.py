from jax.config import config

config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import optax
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import jax
from circuit_layers import (
    get_parameters,
    layer_1,
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

X_train, X_test, y_train, y_test, dimension = UCI_energy_efficiency()

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
num_layers = 2
num_qubits = 6

thetas, num_qubits = get_parameters(
    layer=7,
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


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(parameters, x):
    layer(parameters, x)
    return qml.expval(qml.PauliZ(wires=0))


def layer(parameters, x):
    layer_7(parameters, x, num_layers, num_qubits)


def cost(parameters, x, y):
    return jnp.asarray((circuit(parameters, x) - y) ** 2).reshape(())


def map_cost(parameters, x, y):
    return jnp.sqrt(jnp.mean(jax.vmap(cost, in_axes=[None, 0, 0])(parameters, x, y)))


def predict_one(parameters, x):
    return circuit(parameters, x)


def predict_all(parameters, x):
    return jax.vmap(predict_one, in_axes=[None, 0])(parameters, x)


def map_calculate_metrics(parameters, x, y):
    """
    Calculate rmse and mae of the data
    :param parameters:
    :param x:
    Scaled X
    :param y:
    Scaled y
    :return:
    Root Mean Squared Error, Mean Absolute Error
    """
    y = y.ravel()
    y_pred = predict_all(parameters, x)
    se = (y_pred - y) ** 2
    ae = abs(y_pred - y)
    rmse = jnp.sqrt(jnp.mean(se)).tolist()
    mae = jnp.mean(ae).tolist()
    return rmse, mae


rmse_mae_training = [map_calculate_metrics(thetas, X_train_scaled, y_train_scaled)]
rmse_mae_testing = [map_calculate_metrics(thetas, X_test_scaled, y_test_scaled)]

print(
    "RMSE:",
    round(rmse_mae_training[0][0], 3),
    "MAE:",
    round(rmse_mae_training[0][1], 3),
    "on train data before training",
)

print(
    "RMSE:",
    round(rmse_mae_testing[0][0], 3),
    "MAE:",
    round(rmse_mae_testing[0][1], 3),
    "on test data before training",
)

training_costs = []


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        cost_, grads = jax.value_and_grad(map_cost)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, cost_

    for i in range(epochs):
        x_batches, y_batches = iterate_batches(
            X_train_scaled, y_train_scaled, batch_size
        )
        epoch_cost = []
        for x_batch, y_batch in zip(x_batches, y_batches):
            params, opt_state, cost_ = step(params, opt_state, x_batch, y_batch)
            epoch_cost.append(cost_)
        training_costs.append(jnp.mean(jnp.asarray(epoch_cost)))

    return params


optimizer = optax.adam(learning_rate)
thetas = fit(thetas, optimizer)

trained_parameters = thetas.copy()

rmse_mae_training.append(map_calculate_metrics(thetas, X_train_scaled, y_train_scaled))
rmse_mae_testing.append(map_calculate_metrics(thetas, X_test_scaled, y_test_scaled))
print(
    "RMSE:",
    round(rmse_mae_training[-1][0], 3),
    "MAE:",
    round(rmse_mae_training[-1][1], 3),
    "on train data before training",
)

print(
    "RMSE:",
    round(rmse_mae_testing[-1][0], 3),
    "MAE:",
    round(rmse_mae_testing[-1][1], 3),
    "on test data before training",
)

plt.plot(training_costs)
plt.show()

dev_ = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev_, expansion_strategy="device")
def circuit(parameters, x):
    layer(parameters, x)
    return qml.expval(qml.PauliZ(wires=0))


fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(thetas, X_train_scaled[0])
fig.show()

resources = qml.specs(circuit)(thetas, X_train_scaled[0])
print(resources)

gate_types = dict(resources["gate_sizes"])
total_gates = resources["num_operations"]
depth = resources["depth"]
num_params = thetas.ravel().shape[0]

"""
import pickle

name = 'polynomial_type1_variational'
save_data = []
save_data.extend(
    [num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters.numpy(),
     trained_parameters, training_costs, rmse_mae_training, rmse_mae_testing])
with open(f'new_data_/Regression/Variational/{name}_{num_qubits}_{num_layers}.pkl', 'wb') as file:
    pickle.dump(save_data, file)
"""
