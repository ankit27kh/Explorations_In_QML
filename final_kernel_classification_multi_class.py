from jax.config import config

config.update("jax_enable_x64", True)
from matplotlib import pyplot as plt
import optax
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import jax
from sklearn.svm import SVC
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

seed = 42

X_train, X_test, y_train, y_test, dimension = UCI_heart()
dataset_ = "UCI_heart"
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
num_layers = 3
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

dev = qml.device("default.qubit.jax", wires=num_qubits, shots=num_shots)

projector = jnp.zeros((2 ** num_qubits, 2 ** num_qubits))
projector = projector.at[0, 0].set(1)


def get_one_vs_rest_labels(labels):
    new_label_sets = np.zeros([classes, len(labels)], dtype=int)
    for i in range(classes):
        new_label_sets[i] = [-1 if j == i else 1 for j in labels]
    return new_label_sets


@jax.jit
@qml.qnode(dev, interface="jax")
def kernel(parameters, x1, x2):
    layer(parameters, x1)
    qml.Barrier(wires=range(num_qubits))
    qml.adjoint(layer)(parameters, x2)
    return qml.expval(qml.Hermitian(projector.tolist(), wires=range(num_qubits)))


def layer(parameters, x):
    layer_5(parameters, x, num_layers, num_qubits)


def individual_kernel(parameters, x, y):
    return kernel(parameters, x, y)


mapped_kernel = jax.vmap(
    jax.vmap(individual_kernel, in_axes=(None, None, 0)), in_axes=(None, 0, None)
)


def mapped_kernel_params(params):
    return lambda X, Y: mapped_kernel(params, X, Y)


def target_alignment(X, Y, parameters):
    Kx = mapped_kernel_params(parameters)(X, X)
    _Y = jnp.array(Y)

    # Rescaling
    class_1 = jnp.count_nonzero(_Y == 1)
    class_0 = jnp.count_nonzero(_Y == -1)
    _Y = jnp.where(_Y == 1, _Y / class_1, _Y / class_0)

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


print("Scores before training:")
new_y_train = get_one_vs_rest_labels(y_train)
all_probs_train = np.zeros([classes, len(y_train)])
all_probs_test = np.zeros([classes, len(y_test)])
for i in range(classes):
    svm = SVC(kernel="precomputed", probability=True, random_state=seed)

    training_kerneL_matrix = mapped_kernel(
        all_thetas[i], X_train_scaled, X_train_scaled
    )
    testing_kernel_matrix = mapped_kernel(all_thetas[i], X_test_scaled, X_train_scaled)
    training_kerneL_matrix = (training_kerneL_matrix + training_kerneL_matrix.T) / 2

    svm.fit(training_kerneL_matrix, new_y_train[i])
    all_probs_train[i] = svm.predict_proba(training_kerneL_matrix)[:, 0]
    all_probs_test[i] = svm.predict_proba(testing_kernel_matrix)[:, 0]
y_predict_train = np.argmax(all_probs_train, axis=0)
y_predict_test = np.argmax(all_probs_test, axis=0)
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

all_ktas = []


def fit(params, optimizer):
    @jax.jit
    def step(params, opt_state, X, Y):
        loss, grads = jax.value_and_grad(cost)(params, X, Y)
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
                epoch_cost.append(-loss)
            scores.append(jnp.mean(jnp.asarray(epoch_cost)))
        all_ktas.append(scores)

    return params


optimizer = optax.adam(learning_rate)
all_thetas = fit(all_thetas, optimizer)

trained_parameters = all_thetas.copy()

print("Scores after training:")
new_y_train = get_one_vs_rest_labels(y_train)
all_probs_train = np.zeros([classes, len(y_train)])
all_probs_test = np.zeros([classes, len(y_test)])
for i in range(classes):
    svm = SVC(kernel="precomputed", probability=True, random_state=seed)

    training_kerneL_matrix = mapped_kernel(
        all_thetas[i], X_train_scaled, X_train_scaled
    )
    testing_kernel_matrix = mapped_kernel(all_thetas[i], X_test_scaled, X_train_scaled)
    training_kerneL_matrix = (training_kerneL_matrix + training_kerneL_matrix.T) / 2

    svm.fit(training_kerneL_matrix, new_y_train[i])
    all_probs_train[i] = svm.predict_proba(training_kerneL_matrix)[:, 0]
    all_probs_test[i] = svm.predict_proba(testing_kernel_matrix)[:, 0]
y_predict_train = np.argmax(all_probs_train, axis=0)
y_predict_test = np.argmax(all_probs_test, axis=0)
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

for scores in all_ktas:
    plt.plot(scores)
plt.show()

dev_ = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev_, expansion_strategy="device")
def kernel(parameters, x1, x2):
    layer(parameters, x1)
    qml.Barrier(wires=range(num_qubits))
    qml.adjoint(layer)(parameters, x2)
    return qml.expval(qml.Hermitian(projector.tolist(), wires=range(num_qubits)))


fig, ax = qml.draw_mpl(kernel, expansion_strategy="device")(
    all_thetas[0], X_train_scaled[0], X_train_scaled[0]
)
fig.show()

resources = qml.specs(kernel)(all_thetas[0], X_train_scaled[0], X_train_scaled[0])
print(resources)

gate_types = dict(resources["gate_sizes"])
total_gates = resources["num_operations"]
depth = resources["depth"]
num_params = thetas.ravel().shape[0] * classes

"""
import pickle

name = f'{dataset_}_type1_kernel'
save_data = []
save_data.extend(
    [num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters,
     trained_parameters, all_ktas, before_train_classification_report,
     before_test_classification_report, after_train_classification_report,
     after_test_classification_report])
with open(f'new_data_/Classification/Kernel/{name}_{num_qubits}_{num_layers}.pkl', 'wb') as file:
    pickle.dump(save_data, file)
"""
