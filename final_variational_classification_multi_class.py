from jax.config import config

config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
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
    concentric_circles,
    multiple_bands,
    four_circles,
    UCI_iris,
    MNIST,
    UCI_abalone,
    UCI_car,
    UCI_heart,
    UCI_wine,
)
import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp

X_train, X_test, y_train, y_test, dimension = concentric_circles()
classes = len(np.unique(y_test))

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
num_layers = 5
num_qubits = 1

all_thetas = []
for i in range(classes):
    thetas, num_qubits = get_parameters(
        layer=5,
        dimension=dimension,
        num_layers=num_layers,
        num_qubits=num_qubits,
    )
    all_thetas.append(thetas.copy())
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


original_parameters = all_thetas.copy()
eps = 0.0001

dev = qml.device("default.qubit.jax", wires=num_qubits, shots=num_shots)


def get_one_vs_rest_labels(labels):
    new_label_sets = np.zeros([classes, len(labels)], dtype=int)
    for i in range(classes):
        new_label_sets[i] = [0 if j == i else 1 for j in labels]
    return new_label_sets


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(parameters, x):
    layer(parameters, x)
    return qml.probs(wires=0)


def layer(parameters, x):
    layer_5(parameters, x, num_layers, num_qubits)


def cost(parameters, x, y):
    p = circuit(parameters, x)
    p0 = jnp.max(jnp.array([eps, p[0]]))
    p1 = jnp.max(jnp.array([eps, p[1]]))
    return jnp.asarray(y * jnp.log(p1) + (1 - y) * jnp.log(p0))


def map_cost(parameters, x, y):
    return -(jnp.mean(jax.vmap(cost, in_axes=[None, 0, 0])(parameters, x, y)))


def predict_one(x, parameters):
    return circuit(parameters, x)[0]


def predict_all(x, parameters):
    all_fids = []
    for i in range(classes):
        all_fids.append(jax.vmap(predict_one, in_axes=[0, None])(x, parameters[i]))
    return np.argmax(all_fids, axis=0)


print("Scores before training:")
y_predict_train = predict_all(X_train_scaled, all_thetas)
y_predict_test = predict_all(X_test_scaled, all_thetas)
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

all_scores = []
new_y_train = get_one_vs_rest_labels(y_train)


def fit(params, optimizer):
    @jax.jit
    def step(params, opt_state, X, Y):
        loss, grads = jax.value_and_grad(map_cost)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for j in range(classes):
        opt_state = optimizer.init(params[j])
        scores = []
        for i in range(epochs):
            x_batches, y_batches = iterate_batches(
                X_train_scaled, new_y_train[j], batch_size
            )
            epoch_cost = []
            for x_batch, y_batch in zip(x_batches, y_batches):
                params[j], opt_state, loss = step(
                    params[j], opt_state, x_batch, y_batch
                )
                epoch_cost.append(loss)
            scores.append(jnp.mean(jnp.asarray(epoch_cost)))

        all_scores.append(scores)

    return params


optimizer = optax.adam(learning_rate)
all_thetas = fit(all_thetas, optimizer)

trained_parameters = all_thetas.copy()

print("Scores after training:")
y_predict_train = predict_all(X_train_scaled, all_thetas)
y_predict_test = predict_all(X_test_scaled, all_thetas)
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
for scores in all_scores:
    plt.plot(scores)
plt.show()

dev_ = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev_, expansion_strategy="device")
def circuit(parameters, x):
    layer(parameters, x)
    return qml.expval(qml.PauliZ(wires=0))


fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(
    all_thetas[0], X_train_scaled[0]
)
fig.show()

resources = qml.specs(circuit)(all_thetas[0], X_train_scaled[0])
print(resources)

gate_types = dict(resources["gate_sizes"])
total_gates = resources["num_operations"]
depth = resources["depth"]
num_params = thetas.ravel().shape[0] * classes
