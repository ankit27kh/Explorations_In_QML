import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".7"
from jax.config import config

config.update("jax_enable_x64", True)
import itertools
import time
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
from jax.scipy.linalg import expm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
import optax

seed = 42
np.random.seed(seed)

p = jnp.pi / 2
steps = 100
num_points = 32
k = 10
num_layers = 5
epochs = 10
lr = 0.01

J = 3

pauli_x = jnp.array([[0, 1], [1, 0]])
pauli_y = jnp.array([[0, -1j], [1j, 0]])
pauli_z = jnp.array([[1, 0], [0, -1]])

num_qubits = int(2 * J)
lost_qubits = 2

if lost_qubits > num_qubits:
    raise ValueError(f'Lost Qubits ({lost_qubits}) must be <= number of qubits ({num_qubits})')
else:
    print('Number of qubits:', num_qubits)
    print('Lost qubits:', lost_qubits)


def get_Ji(n, pauli_i):
    matrix = jnp.zeros([2 ** n, 2 ** n])
    for i in range(1, n + 1):
        matrix = matrix + jnp.kron(
            jnp.identity(2 ** (i - 1)), jnp.kron(pauli_i, jnp.identity(2 ** (n - i)))
        )
    return matrix / 2


def get_J_z_2(n):
    matrix = jnp.zeros([2 ** n, 2 ** n])
    for j in range(2, n + 1):
        for i in range(1, j):
            left = jnp.identity(2 ** (i - 1))
            middle = jnp.identity(2 ** (j - 1 - i))
            right = jnp.identity(2 ** (n - j))
            matrix = matrix + jnp.kron(
                left, jnp.kron(pauli_z, jnp.kron(middle, jnp.kron(pauli_z, right)))
            )
    return matrix / 2


J_x = get_Ji(num_qubits, pauli_x)
J_y = get_Ji(num_qubits, pauli_y)
J_z = get_Ji(num_qubits, pauli_z)
J_z_2 = get_J_z_2(num_qubits)

floquet = expm(-1j * k / 2 / J * J_z_2) @ expm(-1j * p * J_y)

init_theta = jnp.linspace(0, jnp.pi, num_points)
init_phi = jnp.linspace(-jnp.pi, jnp.pi, num_points)

X = jnp.array(list(itertools.product(init_theta, init_phi)))

# Z-Hemispheres
y = jnp.array([-1 if i[0] <= jnp.pi / 2 else 1 for i in X])

# X-Hemispheres
# y = jnp.array([-1 if abs(i[1]) <= np.pi / 2 else 1 for i in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dev = qml.device("default.qubit.jax", wires=num_qubits, shots=None)


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(x, params):
    initial_angles = jnp.array([x[0], x[1], 0])

    qml.broadcast(
        qml.U3,
        wires=range(num_qubits),
        pattern="single",
        parameters=jnp.array([initial_angles] * num_qubits),
    )

    for _ in range(steps):
        qml.QubitUnitary(np.array(floquet), wires=range(num_qubits))

    for i in range(num_layers):
        qml.broadcast(
            qml.U3, wires=range(num_qubits - lost_qubits), pattern="single", parameters=params[i]
        )
        qml.broadcast(qml.CNOT, wires=range(num_qubits - lost_qubits), pattern="ring")

    return qml.expval(qml.PauliZ(0))


weights = np.random.uniform(-jnp.pi, jnp.pi, [num_layers, num_qubits - lost_qubits, 3])
weights = jnp.array(weights)

original_parameters = weights.copy()


def cost(parameters, x, y):
    p = circuit(x, parameters)
    return (y - p) ** 2


def map_cost(parameters, x, y):
    return jnp.mean(jax.vmap(cost, in_axes=[None, 0, 0])(parameters, x, y))


def predict_one(x, parameters):
    ep = circuit(x, parameters)
    return 2 * (ep >= 0) - 1


def predict_all(x, parameters):
    return jax.vmap(predict_one, in_axes=[0, None])(x, parameters)


train_costs = []
test_costs = []


def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X, Y):
        grads = jax.grad(map_cost)(params, X, Y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for i in range(epochs):
        params, opt_state = step(params, opt_state, X_train, y_train)
        train_costs.append(map_cost(params, X_train, y_train))
        test_costs.append(map_cost(params, X_test, y_test))
        test_acc_score_epoch = accuracy_score(y_pred=predict_all(X_test, params), y_true=y_test)
        # print(test_acc_score_epoch)
        if test_acc_score_epoch >= .844:
            print(i + 1, 'epoch')
            break

    return params


print("Scores before training:")
y_predict_train = predict_all(X_train, weights)
y_predict_test = predict_all(X_test, weights)
print(
    "Training Data:",
    accuracy_score(y_pred=y_predict_train, y_true=y_train),
    "Testing Data:",
    accuracy_score(y_pred=y_predict_test, y_true=y_test),
)

optimizer = optax.adam(lr)
t1 = time.time()
weights = fit(weights, optimizer)
print(time.time() - t1)

trained_parameters = weights.copy()

plt.plot(train_costs)
plt.plot(test_costs)
plt.legend(["Train", "Test"])
plt.show()

print("Scores after training:")
y_predict_train = predict_all(X_train, weights)
y_predict_test = predict_all(X_test, weights)
train_acc = accuracy_score(y_pred=y_predict_train, y_true=y_train)
test_acc = accuracy_score(y_pred=y_predict_test, y_true=y_test)
print(
    "Training Data:",
    train_acc,
    "Testing Data:",
    test_acc,
)

dev_ = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev_)
def circuit(x, params):
    initial_angles = jnp.array([x[0], x[1], 0])

    qml.broadcast(
        qml.U3,
        wires=range(num_qubits),
        pattern="single",
        parameters=jnp.array([initial_angles] * num_qubits),
    )

    for _ in range(steps):
        qml.QubitUnitary(np.array(floquet), wires=range(num_qubits))

    for i in range(num_layers):
        qml.broadcast(
            qml.U3, wires=range(num_qubits - lost_qubits), pattern="single", parameters=params[i]
        )
        qml.broadcast(qml.CNOT, wires=range(num_qubits - lost_qubits), pattern="ring")

    return qml.expval(qml.PauliZ(0))


"""
fig, ax = qml.draw_mpl(circuit)(X_train[0], weights)
fig.show()
"""
resources = qml.specs(circuit)(X_train[0], weights)
# print(resources)

gate_types = dict(resources["gate_sizes"])
total_gates = resources["num_operations"]
depth = resources["depth"]
num_params = weights.ravel().shape[0]
"""
import pickle

save_data = []
save_data.extend(
    [
        num_layers,
        num_qubits,
        num_params,
        gate_types,
        total_gates,
        depth,
        resources,
        original_parameters,
        trained_parameters,
        train_costs,
        test_costs,
        train_acc,
        test_acc,
        lost_qubits,
    ]
)
with open(
        f"new_data_/rotor/lost_qubits/rotor_{J}_{num_layers}_{steps}_{k}_{lost_qubits}.pkl", "wb"
) as file:
    pickle.dump(save_data, file)
print("finished=", J, k, steps, num_layers, lost_qubits)
"""
import seaborn as sns

sns.set_theme()

plt.plot(range(7), [.844, .744, .620, .588, .491, .367, .711], ":.")
plt.ylabel('Validation Accuracy')
plt.xlabel('Lost Qubits')
# plt.savefig('new_data_/rotor/lost_qubits/lost_qubits_scores.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
