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

"""
k=10,steps=1000,epochs=1000,layers=1
J=1 -> 92.5%
J=1.5 -> 86.7%
J=2 -> 90.9%
J=2.5 -> 97.4%
J=3 -> 86.0%
"""

p = jnp.pi / 2
steps = 10
num_points = 32
k = 10
num_layers = 1
epochs = 10
lr = 0.01

J = 1

pauli_x = jnp.array([[0, 1], [1, 0]])
pauli_y = jnp.array([[0, -1j], [1j, 0]])
pauli_z = jnp.array([[1, 0], [0, -1]])

num_qubits = int(2 * J)


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
            qml.U3, wires=range(num_qubits), pattern="single", parameters=params[i]
        )
        qml.broadcast(qml.CNOT, wires=range(num_qubits), pattern="ring")

    return qml.expval(qml.PauliZ(0))


weights = np.random.uniform(-jnp.pi, jnp.pi, [num_layers, num_qubits, 3])
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

    return params


"""
print("Scores before training:")
y_predict_train = predict_all(X_train, weights)
y_predict_test = predict_all(X_test, weights)
print(
    "Training Data:",
    accuracy_score(y_pred=y_predict_train, y_true=y_train),
    "Testing Data:",
    accuracy_score(y_pred=y_predict_test, y_true=y_test),
)
"""

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
print(
    "Training Data:",
    accuracy_score(y_pred=y_predict_train, y_true=y_train),
    "Testing Data:",
    accuracy_score(y_pred=y_predict_test, y_true=y_test),
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
            qml.U3, wires=range(num_qubits), pattern="single", parameters=params[i]
        )
        qml.broadcast(qml.CNOT, wires=range(num_qubits), pattern="ring")

    return qml.expval(qml.PauliZ(0))


fig, ax = qml.draw_mpl(circuit)(X_train[0], weights)
resources = qml.specs(circuit)(X_train[0], weights)
print(resources)

gate_types = dict(resources["gate_sizes"])
total_gates = resources["num_operations"]
depth = resources["depth"]
num_params = weights.ravel().shape[0]
