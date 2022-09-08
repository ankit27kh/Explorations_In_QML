import itertools
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import seaborn as sns

sns.set_theme()

directory = "new_data_/rotor"


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


i = 0
files = 1980
jj = 1
"""
[num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters,
 trained_parameters, train_costs, test_costs, train_acc, test_acc]"""

all_original_parameters = []
all_trained_parameters = []
all_accuracy_after_training = []
all_accuracy_after_testing = []
all_layers = []
all_qubits = []
all_num_params = []
all_gate_types = []
all_num_gates = []
all_depths = []
all_resources = []
all_train_costs = []
all_test_costs = []
all_J = []
all_steps = []
all_k = []
for file in sorted_alphanumeric(os.listdir(directory)):
    if file.split(".")[-1] != "pkl":
        continue
    path = os.path.join(directory, file)
    with open(path, "rb") as f:
        data = pickle.load(f)
        legend = file.split("_")

        j = legend[1]
        n = legend[3]
        k = legend[-1].rstrip(".pkl")

        j, n, k = float(j), int(n), float(k)

        if j != jj:
            continue

        all_J.append(j)
        all_steps.append(n)
        all_k.append(k)

        all_layers.append(data[0])
        all_qubits.append(data[1])
        all_num_params.append(data[2])
        all_gate_types.append(data[3])
        all_num_gates.append(data[4])
        all_depths.append(data[5])
        all_resources.append(data[6])
        all_original_parameters.append(data[7])
        all_trained_parameters.append(data[8])
        all_train_costs.append(data[9])
        all_test_costs.append(data[10])
        all_accuracy_after_training.append(data[11])
        all_accuracy_after_testing.append(data[12])

    i = i + 1
    if i == files:
        break

df = pd.DataFrame(
    {
        "Layers": all_layers,
        "Steps": all_steps,
        "k": all_k,
        "num_qubits": all_qubits,
        "num_params": all_num_params,
        "gate_types": all_gate_types,
        "depths": all_depths,
        "acc_train": all_accuracy_after_training,
        "num_gates": all_num_gates,
        "resources": all_resources,
        "org_para": all_original_parameters,
        "train_para": all_trained_parameters,
        "acc_test": all_accuracy_after_testing,
        "cost_train": all_train_costs,
        "cost_test": all_test_costs,
    }
)

df_k0_01 = df[df["k"] == 0.01]
df_k0_1 = df[df["k"] == 0.1]
df_k1 = df[df["k"] == 1]
df_k2 = df[df["k"] == 2]
df_k3 = df[df["k"] == 3]
df_k4 = df[df["k"] == 4]
df_k5 = df[df["k"] == 5]
df_k6 = df[df["k"] == 6]
df_k10 = df[df["k"] == 10]
df_k50 = df[df["k"] == 50]
df_k100 = df[df["k"] == 100]

min_acc = []
max_acc = []
for data_k in [
    df_k0_01,
    df_k0_1,
    df_k1,
    df_k2,
    df_k3,
    df_k4,
    df_k5,
    df_k6,
    df_k10,
    df_k50,
    df_k100,
]:
    data_s1 = data_k[data_k["Steps"] == 1].sort_values(by="Layers")
    data_s10 = data_k[data_k["Steps"] == 10].sort_values(by="Layers")
    data_s20 = data_k[data_k["Steps"] == 20].sort_values(by="Layers")
    data_s50 = data_k[data_k["Steps"] == 50].sort_values(by="Layers")
    data_s100 = data_k[data_k["Steps"] == 100].sort_values(by="Layers")
    data_s1000 = data_k[data_k["Steps"] == 1000].sort_values(by="Layers")

    min_acc.append(min(data_k["acc_test"]))
    max_acc.append(max(data_k["acc_test"]))

    """for i, data_s in enumerate([data_s1, data_s10, data_s20, data_s50, data_s100, data_s1000]):
        data = np.asarray(data_s['acc_train']).T
        plt.plot(range(i * 6, 6 * (i + 1)), data, ":.")
    plt.xticks(list(range(36)), labels=[1, 2, 3, 4, 5, 10] * 6)
    plt.title("Training Data, k=" + str(data_k.k.to_list()[0]))
    plt.ylabel("Accuracy")
    for x in range(5, 36, 6):
        plt.axvline(x, linestyle=":")
    plt.show()"""
    plt.figure(figsize=(8, 5))
    for i, data_s in enumerate(
        [data_s1, data_s10, data_s20, data_s50, data_s100, data_s1000]
    ):
        data = np.asarray(data_s["acc_test"]).T
        plt.plot(range(i * 6, 6 * (i + 1)), data, ":.")
    plt.xticks(list(range(36)), labels=[1, 2, 3, 4, 5, 10] * 6)
    # plt.title("k=" + ("%f" % float(str(data_k.k.to_list()[0]))).rstrip("0").rstrip("."))
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Number of Layers")
    for x in range(5, 36, 6):
        plt.axvline(x, linestyle=":")
    plt.text(
        3 / 39,
        0.05,
        "n=1",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="red", alpha=0.5, boxstyle="round"),
        style="italic",
    )
    plt.text(
        8.5 / 39,
        0.05,
        "n=10",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="red", alpha=0.5, boxstyle="round"),
        style="italic",
    )
    plt.text(
        14.5 / 39,
        0.05,
        "n=20",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="red", alpha=0.5, boxstyle="round"),
        style="italic",
    )
    plt.text(
        20.5 / 39,
        0.05,
        "n=50",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="red", alpha=0.5, boxstyle="round"),
        style="italic",
    )
    plt.text(
        26.4 / 39,
        0.05,
        "n=100",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="red", alpha=0.5, boxstyle="round"),
        style="italic",
    )
    plt.text(
        32 / 39,
        0.05,
        "n=1000",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="red", alpha=0.5, boxstyle="round"),
        style="italic",
    )
    # plt.savefig(f'new_data_/rotor/plots/results_{jj}_{str(data_k.k.to_list()[0])}.pdf', format='pdf', dpi=1200,
    #            bbox_inches='tight')
    plt.show()

"""plt.plot(min_acc, ':.', label='Minimum Values')
plt.plot(max_acc, ':*', label='Maximum Values')
plt.legend()
plt.xticks(range(11), [0.01, 0.1, 1, 2, 3, 4, 5, 6, 10, 50, 100])
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.show()

print(min(max_acc))"""

"""
k=10,steps=1000,epochs=1000,layers=1
J=1 -> 92.5%
J=1.5 -> 86.7%
J=2 -> 90.9%
J=2.5 -> 97.4%
J=3 -> 86.0%
"""
J = [1, 1.5, 2, 2.5, 3]
acc = [0.925, 0.867, 0.909, 0.974, 0.86]
plt.plot(J, acc, ":*")
plt.xlabel("J")
plt.xticks(J, J)
plt.ylabel("Validation Accuracy")
# plt.savefig('new_data_/rotor/plots/single_layer.eps', format='eps', dpi=1200, bbox_inches='tight')
plt.show()

j1_max = [
    0.9675324675324676,
    0.9837662337662337,
    0.9512987012987013,
    0.974025974025974,
    0.9577922077922078,
    0.9577922077922078,
    0.9642857142857143,
    0.9772727272727273,
    0.9707792207792207,
    0.9577922077922078,
    0.9383116883116883,
]
j1_min = [
    0.07792207792207792,
    0.16558441558441558,
    0.11363636363636363,
    0.05519480519480519,
    0.1038961038961039,
    0.13636363636363635,
    0.17857142857142858,
    0.2305194805194805,
    0.04220779220779221,
    0.22402597402597402,
    0.09090909090909091,
]
j1_5_max = [
    0.9675324675324676,
    0.987012987012987,
    0.961038961038961,
    0.961038961038961,
    0.9675324675324676,
    0.9577922077922078,
    0.9772727272727273,
    0.987012987012987,
    0.9577922077922078,
    0.9772727272727273,
    0.9772727272727273,
]
j1_5_min = [
    0.4577922077922078,
    0.24675324675324675,
    0.19155844155844157,
    0.3181818181818182,
    0.2435064935064935,
    0.275974025974026,
    0.3409090909090909,
    0.33116883116883117,
    0.18506493506493507,
    0.31493506493506496,
    0.4707792207792208,
]
j2_max = [
    0.9707792207792207,
    0.9512987012987013,
    0.9675324675324676,
    0.9805194805194806,
    0.9675324675324676,
    0.9642857142857143,
    0.948051948051948,
    0.9967532467532467,
    0.948051948051948,
    0.9902597402597403,
    0.9642857142857143,
]
j2_min = [
    0.2012987012987013,
    0.18506493506493507,
    0.33766233766233766,
    0.2824675324675325,
    0.09415584415584416,
    0.10064935064935066,
    0.36038961038961037,
    0.08116883116883117,
    0.1461038961038961,
    0.1590909090909091,
    0.237012987012987,
]
j2_5_max = [
    0.9837662337662337,
    0.9577922077922078,
    0.9675324675324676,
    0.9675324675324676,
    0.9642857142857143,
    0.9805194805194806,
    0.9935064935064936,
    0.9707792207792207,
    0.9805194805194806,
    0.9707792207792207,
    0.9675324675324676,
]
j2_5_min = [
    0.21428571428571427,
    0.42857142857142855,
    0.3474025974025974,
    0.36688311688311687,
    0.3181818181818182,
    0.4155844155844156,
    0.4772727272727273,
    0.474025974025974,
    0.3961038961038961,
    0.4642857142857143,
    0.38636363636363635,
]
j3_max = [
    0.9805194805194806,
    0.961038961038961,
    0.974025974025974,
    0.9707792207792207,
    0.9772727272727273,
    0.9642857142857143,
    0.9415584415584416,
    0.974025974025974,
    0.9675324675324676,
    0.974025974025974,
    0.974025974025974,
]
j3_min = [
    0.3474025974025974,
    0.275974025974026,
    0.22402597402597402,
    0.37337662337662336,
    0.3474025974025974,
    0.18506493506493507,
    0.2922077922077922,
    0.3246753246753247,
    0.5,
    0.525974025974026,
    0.38311688311688313,
]

plt.plot(j1_max, "r:.", label="J=1")
plt.plot(j1_min, "r:.")
plt.plot(j1_5_max, "b:.", label="J=1.5")
plt.plot(j1_5_min, "b:.")
plt.plot(j2_max, "g:.", label="J=2")
plt.plot(j2_min, "g:.")
plt.plot(j2_5_max, "m:.", label="J=2.5")
plt.plot(j2_5_min, "m:.")
plt.plot(j3_max, "c:.", label="J=3")
plt.plot(j3_min, "c:.")
plt.xticks(range(11), [0.01, 0.1, 1, 2, 3, 4, 5, 6, 10, 50, 100])
plt.ylabel("Validation Accuracy")
plt.xlabel("k")
plt.axhline(y=0.7, linestyle="--")
plt.text(
    0,
    0.7,
    "Maximum Values",
    transform=plt.gca().transAxes,
)
plt.text(
    0,
    0.62,
    "Minimum Values",
    transform=plt.gca().transAxes,
)
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.subplots_adjust(right=0.8)
# plt.savefig('new_data_/rotor/plots/min_max.eps', format='eps', dpi=1200, bbox_inches='tight')
plt.show()

# Make Sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
x1 = r * sin(phi) * cos(theta)
y1 = r * sin(phi) * sin(theta)
z1 = r * cos(phi)

init_theta = np.linspace(0, np.pi, 32)
init_phi = np.linspace(-np.pi, np.pi, 32)

X = np.array(list(itertools.product(init_theta, init_phi)))
y = np.array([-1 if i[0] <= np.pi / 2 else 1 for i in X])

xx = sin(X[:, 0]) * cos(X[:, 1])
yy = sin(X[:, 0]) * sin(X[:, 1])
zz = cos(X[:, 0])

# Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, color="c", alpha=0.3, linewidth=0)

"""for i in range(32 ** 2):
    if y[i] == -1:
        ax.scatter(xx[i], yy[i], zz[i], color="k", s=20, marker='o', label='Label 1')
    else:
        ax.scatter(xx[i], yy[i], zz[i], color="k", s=20, marker='o', label='Label -1')"""
ax.scatter(xx, yy, zz, color="k", s=20)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect((1, 1, 1))
# ax.axis('off')
ax.set_xticks([-1, 0, 1], [-1, 0, 1])
ax.set_yticks([-1, 0, 1], [-1, 0, 1])
ax.set_zticks([-1, 0, 1], [-1, 0, 1])
# plt.title("Initial Points Used")
# legend = ax.get_legend_handles_labels()
# plt.legend(handles=[legend[0][0], legend[0][-1]], labels=['Label 1', 'Lable -1'])
plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.1, wspace=0, hspace=0)
# plt.savefig('new_data_/rotor/plots/input_points.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
