import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.ibmq.runtime import UserMessenger
from scipy.linalg import expm
import qiskit.algorithms.optimizers


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])


def get_Ji(n, pauli_i):
    matrix = np.zeros([2 ** n, 2 ** n])
    for i in range(1, n + 1):
        matrix = matrix + np.kron(
            np.identity(2 ** (i - 1)), np.kron(pauli_i, np.identity(2 ** (n - i)))
        )
    return matrix / 2


def get_J_z_2(n):
    matrix = np.zeros([2 ** n, 2 ** n])
    for j in range(2, n + 1):
        for i in range(1, j):
            left = np.identity(2 ** (i - 1))
            middle = np.identity(2 ** (j - 1 - i))
            right = np.identity(2 ** (n - j))
            matrix = matrix + np.kron(
                left, np.kron(pauli_z, np.kron(middle, np.kron(pauli_z, right)))
            )
    return matrix / 2


def get_operators(n, k, J, p):
    J_x = get_Ji(n, pauli_x)
    J_y = get_Ji(n, pauli_y)
    J_z = get_Ji(n, pauli_z)
    J_z_2 = get_J_z_2(n)

    floquet = expm(-1j * k / 2 / J * J_z_2) @ expm(-1j * p * J_y)
    return J_x, J_y, J_z, J_z_2, floquet




def create_circuit(n, x, layers, steps, param, floquet, J_x, J_y):
    qc = QuantumCircuit(n, 1)

    qc.u(x[0], x[1], 0, qc.qubits)

    for _ in range(steps):
        qc.unitary(floquet, range(n))

    for i in range(layers):
        for j in range(n):
            qc.u(param[i][j][0], param[i][j][1], param[i][j][2], j)
        for i, j in zip(range(n - 1), range(1, n)):
            qc.cx(i, j)
        if n > 2:
            qc.cx(n - 1, 0)
    qc.measure(0, 0)
    return qc


def get_expectation_values(
        n, X, layers, steps, params, backend, shots, floquet, J_x, J_y, seed
):
    all_circuits = []
    for x in X:
        qc = create_circuit(n, x, layers, steps, params, floquet, J_x, J_y)
        qc = transpile(qc, backend=backend, optimization_level=0, seed_transpiler=seed)
        all_circuits.append(qc)

    all_circuits = transpile(
        all_circuits, backend=backend, optimization_level=0, seed_transpiler=seed
    )
    circuit_groups = list(split(all_circuits, int(np.ceil(len(all_circuits) / 300))))

    results = []
    for circuits in circuit_groups:
        res = backend.run(circuits, shots=shots).result()
        results.append(res)

    expectation_values = []
    for group in results:
        counts = group.get_counts()
        expectation_values.extend(
            [(i.get("1", 0) - i.get("0", 0)) / shots for i in counts]
        )

    return np.array(expectation_values)


def get_predictions(exp_values):
    return 2 * (exp_values >= 0) - 1


def get_cost(params, n, X, layers, steps, backend, shots, y, floquet, J_x, J_y, seed):
    params = params.reshape([layers, n, 3])
    exp_values = get_expectation_values(
        n, X, layers, steps, params, backend, shots, floquet, J_x, J_y, seed
    )
    return np.mean(np.square(y - exp_values))


def main(
        backend,
        user_messenger,
        X_train,
        y_train,
        X_test,
        J,
        layers,
        steps,
        params,
        k,
        max_iter,
        p=np.pi / 2,
        seed=42,
        shots=1000,
):
    np.random.seed(seed)
    qiskit.utils.algorithm_globals.random_seed = seed
    n = int(2 * J)

    J_x, J_y, J_z, J_z_2, floquet = get_operators(n, k, J, p)

    output = {}
    output["INPUTS"] = {
        "Backend": backend.name(),
        "seed": seed,
        "layers": layers,
        "J": J,
        "p": p,
        "steps": steps,
        "k": k,
        "max_iter": max_iter,
        "shots": shots,
    }
    params = params.ravel()

    opt = qiskit.algorithms.optimizers.SPSA(
        maxiter=max_iter,
        learning_rate=0.1,
        perturbation=0.2,
        # second_order=True,
        # blocking=True,
    )
    t1 = time.time()
    res = opt.optimize(
        params.ravel().shape[0],
        lambda params: get_cost(
            params,
            n,
            X_train,
            layers,
            steps,
            backend,
            shots,
            y_train,
            floquet,
            J_x,
            J_y,
            seed,
        ),
        initial_point=params.ravel(),
    )
    output["Optimisation Time"] = time.time() - t1

    output["OUT"] = res
    updated_params = res[0]
    updated_params = updated_params.reshape([layers, n, 3])
    t1 = time.time()
    exp_values_train = get_expectation_values(
        n,
        X_train,
        layers,
        steps,
        updated_params,
        backend,
        shots,
        floquet,
        J_x,
        J_y,
        seed,
    )
    exp_values_test = get_expectation_values(
        n,
        X_test,
        layers,
        steps,
        updated_params,
        backend,
        shots,
        floquet,
        J_x,
        J_y,
        seed,
    )
    output["Prediction Time"] = time.time() - t1
    y_pred_train = get_predictions(exp_values_train)
    y_pred_test = get_predictions(exp_values_test)
    output["Training Predictions"] = y_pred_train
    output["Testing Predictions"] = y_pred_test
    return output


msg = UserMessenger()
