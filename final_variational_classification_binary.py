from jax.config import config

config.update("jax_enable_x64", True)
from matplotlib import pyplot as plt
import optax
from sklearn.metrics import classification_report
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
    circle,
    checkerboard,
    moons,
    UCI_wine_quality_classification,
    UCI_breast_cancer,
    UCI_bank,
    UCI_Adult,
)
import pennylane as qml
import jax.numpy as jnp

X_train, X_test, y_train, y_test, dimension = checkerboard()

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_x_mm = MinMaxScaler([-1, 1])
X_train_scaled = scaler_x_mm.fit_transform(X_train_scaled)
X_test_scaled = scaler_x_mm.transform(X_test_scaled)

X_train_scaled, X_test_scaled, y_train, y_test = (
    jnp.asarray(X_train_scaled),
    jnp.asarray(X_test_scaled),
    jnp.asarray(y_train),
    jnp.asarray(y_test),
)

num_shots = None
num_layers = 1
num_qubits = 2

thetas, num_qubits = get_parameters(
    layer=1,
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
eps = 0.0001

dev = qml.device("default.qubit.jax", wires=num_qubits, shots=num_shots)


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(parameters, x):
    layer(parameters, x)
    return qml.probs(wires=0)


def layer(parameters, x):
    layer_1(parameters, x, num_layers, num_qubits)


def cost(parameters, x, y):
    p = circuit(parameters, x)
    p0 = jnp.max(jnp.array([eps, p[0]]))
    p1 = jnp.max(jnp.array([eps, p[1]]))
    return jnp.asarray(y * jnp.log(p1) + (1 - y) * jnp.log(p0))


def map_cost(parameters, x, y):
    return -(jnp.mean(jax.vmap(cost, in_axes=[None, 0, 0])(parameters, x, y)))


def predict_one(x, parameters):
    return jnp.argmax(circuit(parameters, x))


def predict_all(x, parameters):
    return jax.vmap(predict_one, in_axes=[0, None])(x, parameters)


print("Scores before training:")
y_predict_train = predict_all(X_train_scaled, thetas)
y_predict_test = predict_all(X_test_scaled, thetas)
print("Training Data:")
print(classification_report(y_pred=y_predict_train, y_true=y_train, zero_division=0))
print("Testing Data:")
print(classification_report(y_pred=y_predict_test, y_true=y_test, zero_division=0))
before_train_classification_report = classification_report(
    y_pred=y_predict_train, y_true=y_train, zero_division=0, output_dict=True
)
before_test_classification_report = classification_report(
    y_pred=y_predict_test, y_true=y_test, zero_division=0, output_dict=True
)

scores = []


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        loss, grads = jax.value_and_grad(map_cost)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(epochs):
        x_batches, y_batches = iterate_batches(X_train_scaled, y_train, batch_size)
        epoch_cost = []
        for x_batch, y_batch in zip(x_batches, y_batches):
            params, opt_state, loss = step(params, opt_state, x_batch, y_batch)
            epoch_cost.append(loss)
        scores.append(jnp.mean(jnp.asarray(epoch_cost)))

    return params


optimizer = optax.adam(learning_rate)
thetas = fit(thetas, optimizer)

trained_parameters = thetas.copy()

print("Scores after training:")
y_predict_train = predict_all(X_train_scaled, thetas)
y_predict_test = predict_all(X_test_scaled, thetas)
print("Training Data:")
print(classification_report(y_pred=y_predict_train, y_true=y_train, zero_division=0))
print("Testing Data:")
print(classification_report(y_pred=y_predict_test, y_true=y_test, zero_division=0))
after_train_classification_report = classification_report(
    y_pred=y_predict_train, y_true=y_train, zero_division=0, output_dict=True
)
after_test_classification_report = classification_report(
    y_pred=y_predict_test, y_true=y_test, zero_division=0, output_dict=True
)

plt.plot(scores)
plt.show()

dev_ = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev_, expansion_strategy="device")
def circuit(parameters, x):
    layer(parameters, x)
    return qml.probs(wires=0)


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

name = 'chekerboard_type1_variational'
save_data = []
save_data.extend(
    [num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters.numpy(),
     trained_parameters, scores, before_train_classification_report, before_test_classification_report,
     after_train_classification_report, after_test_classification_report])
with open(f'new_data_/Classification/Variational/{name}_{num_qubits}_{num_layers}.pkl', 'wb') as file:
    pickle.dump(save_data, file)
"""
