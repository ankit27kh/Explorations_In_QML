import pennylane as qml
import pennylane.templates.embeddings as embeddings
import pennylane.templates.layers as layers
import pennylane.numpy as np
import jax.numpy as jnp

seed = 42
np.random.seed(seed)


def get_parameters(layer, dimension, num_layers, num_entangling_layers=1, num_qubits=1):
    if layer in [1, 8]:
        num_qubits = dimension
        return (
            np.random.uniform(
                -np.pi / 2,
                np.pi / 2,
                [num_layers, num_entangling_layers, num_qubits, 3],
            ),
            num_qubits,
        )
    elif layer in [2, 3]:
        if layer == 2:
            if num_qubits % 2 != 0:
                raise ValueError("Num Qubits must be multiple of 2 for layer 2")
            if num_layers == 1:
                raise ValueError("Num Layers must be >1 for layer 2")
        num_qubits = num_qubits
        num_rots = int(np.ceil(dimension / 3))
        return (
            np.random.uniform(
                -np.pi / 2, np.pi / 2, num_layers * num_rots * num_qubits * 3 * 2
            ),
            num_qubits,
        )
    elif layer == 4:
        num_qubits = dimension
        shape = embeddings.QAOAEmbedding.shape(num_layers, num_qubits)
        return np.random.uniform(-np.pi / 2, np.pi / 2, shape), num_qubits
    elif layer in [5, 7]:
        num_qubits = 1
        num_rots = int(np.ceil(dimension / 3))
        return (
            np.random.uniform(-np.pi / 2, np.pi / 2, num_layers * num_rots * 3 * 2),
            num_qubits,
        )
    elif layer == 6:
        num_qubits = 1
        return (
            np.random.uniform(-np.pi / 2, np.pi / 2, [num_layers, dimension, 4]),
            num_qubits,
        )


def layer_1(parameters, x, num_layers, num_qubits, rot="Y", two_gate=qml.CNOT):
    """
    * Angle Embedding -> Wires = Dimension
    * Variational Layer - Entangling Layer -> num_entangling_layers times
    """
    for j in range(num_layers):
        embeddings.AngleEmbedding(x, wires=range(num_qubits), rotation=rot)
        layers.StronglyEntanglingLayers(
            parameters[j], wires=range(num_qubits), imprimitive=two_gate
        )


def layer_2(parameters, x, num_layers, num_qubits):
    # Multi Qubit data reupload with entanglement
    dim = len(x)
    num_rots = int(np.ceil(dim / 3))
    extra = 3 * num_rots - dim
    x = jnp.concatenate([x, jnp.zeros(extra)])
    num_thetas = num_layers * num_rots * num_qubits * 3
    varias = parameters[:num_thetas].reshape([num_layers, num_qubits, num_rots, 3])
    weights = parameters[num_thetas:].reshape([num_layers, num_qubits, num_rots, 3])
    for k in range(num_layers):
        for j in range(num_qubits):
            for i in range(num_rots):
                qml.Rot(
                    *(x[3 * i : 3 * (i + 1)] * weights[k][j][i] + varias[k][j][i]),
                    wires=j
                )
        if k != num_layers - 1 and num_qubits > 1:
            if k % 2 == 0:
                wire = 0
                for _ in range(num_qubits // 2):
                    qml.CZ(wires=[wire, wire + 1])
                    wire = wire + 2
            else:
                wire = 1
                for _ in range(num_qubits // 2 - 1):
                    qml.CZ(wires=[wire, wire + 1])
                    wire = wire + 2
                qml.CZ(wires=[0, num_qubits - 1])


def layer_3(parameters, x, num_layers, num_qubits):
    # Multi Qubit data reupload without entanglement
    dim = len(x)
    num_rots = int(np.ceil(dim / 3))
    extra = 3 * num_rots - dim
    x = jnp.concatenate([x, jnp.zeros(extra)])
    num_thetas = num_layers * num_rots * num_qubits * 3
    varias = parameters[:num_thetas].reshape([num_layers, num_qubits, num_rots, 3])
    weights = parameters[num_thetas:].reshape([num_layers, num_qubits, num_rots, 3])
    for k in range(num_layers):
        for j in range(num_qubits):
            for i in range(num_rots):
                qml.Rot(
                    *(x[3 * i : 3 * (i + 1)] * weights[k][j][i] + varias[k][j][i]),
                    wires=j
                )


def layer_4(parameters, x, num_layers, num_qubits):
    embeddings.QAOAEmbedding(features=x, weights=parameters, wires=range(num_qubits))


def layer_5(parameters, x, num_layers, num_qubits):
    # Single qubit data reupload
    dim = len(x)
    num_rots = int(np.ceil(dim / 3))
    extra = 3 * num_rots - dim
    x = jnp.concatenate([x, jnp.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = parameters[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = parameters[num_thetas:].reshape([num_layers, num_rots, 3])
    for k in range(num_layers):
        for i in range(num_rots):
            qml.Rot(*(x[3 * i : 3 * (i + 1)] * weights[k][i] + varias[k][i]), wires=0)


def layer_6(parameters, x, num_layers, num_qubits):
    """
    Single Qubit Data Re-upload different kind
    * Embedding feature 1 - Variational Layer - Embedding feature 2 - Variational layer and so on
    Feature Embedding is with feature weights
    """
    for k in range(num_layers):
        for i in range(len(x)):
            t1, t2, t3, w = parameters[k][i]
            qml.RX(x[i] + w, wires=0)
            qml.Rot(t1, t2, t3, wires=0)


def layer_7(parameters, x, num_layers, num_qubits):
    # Single qubit data reupload with H
    dim = len(x)
    num_rots = int(np.ceil(dim / 3))
    extra = 3 * num_rots - dim
    x = jnp.concatenate([x, jnp.zeros(extra)])
    num_thetas = num_layers * num_rots * 3
    varias = parameters[:num_thetas].reshape([num_layers, num_rots, 3])
    weights = parameters[num_thetas:].reshape([num_layers, num_rots, 3])
    for k in range(num_layers):
        for i in range(num_rots):
            qml.Rot(*(x[3 * i : 3 * (i + 1)] * weights[k][i] + varias[k][i]), wires=0)
        qml.Hadamard(wires=0)


def layer_8(parameters, x, num_layers, num_qubits, rot="Y", two_gate=qml.CNOT):
    """
    * Angle Embedding -> Wires = Dimension
    * Hadamard Layer
    * Variational Layer - Entangling Layer -> num_entangling_layers times
    """
    for j in range(num_layers):
        embeddings.AngleEmbedding(x, wires=range(num_qubits), rotation=rot)
        qml.broadcast(qml.Hadamard, wires=range(num_qubits), pattern="single")
        layers.StronglyEntanglingLayers(
            parameters[j], wires=range(num_qubits), imprimitive=two_gate
        )


if __name__ == "__main__":
    num_layers = 1
    num_shots = None
    num_qubits = 1
    ll = 5

    x = jnp.array([1, 2, 3])
    y = jnp.array([11, 22, 33])
    thetas, num_qubits = get_parameters(
        layer=ll,
        dimension=len(x),
        num_layers=num_layers,
        num_entangling_layers=1,
        num_qubits=num_qubits,
    )
    dev = qml.device("default.qubit", wires=num_qubits, shots=num_shots)

    @qml.qnode(dev, expansion_strategy="device")
    def circuit(parameters, x1, x2):
        layer_5(parameters, x1, num_layers, num_qubits)
        qml.adjoint(layer_5)(parameters, x2, num_layers, num_qubits)
        return qml.probs(wires=range(num_qubits))

    print(circuit(thetas, y, x)[0])

    fig, ax = qml.draw_mpl(circuit, expansion_strategy="device", fontsize="xx-large")(
        thetas, x, y
    )
    fig.show()
    print(qml.draw(circuit, expansion_strategy="device", max_length=200)(thetas, x, y))
    resources = qml.specs(circuit)(thetas, y, x)
    print(resources)
