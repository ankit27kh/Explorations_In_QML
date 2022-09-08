import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

sns.set_theme()

directory = "new_data_/Regression/Kernel"

"""
[num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters.numpy(),
     trained_parameters, kta_values, before_training_metrics, before_testing_metrics, after_training_metrics,
     after_testing_metrics]
"""

all_original_parameters = []
all_trained_parameters = []

all_layers = []
all_qubits = []
all_num_params = []
all_gate_types = []
all_num_gates = []
all_depths = []
all_resources = []
all_type = []
all_datasets = []

all_metrics_before_testing = []
all_metrics_before_training = []
all_metrics_after_training = []
all_metrics_after_testing = []

all_kta_values = []

all_best_scores_after_test = []
all_best_scores_before_test = []

i = 0
files = 1000

table_rmse_after = []
table_rmse_before = []
table_mae_after = []
table_mae_before = []


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


for file in sorted_alphanumeric(os.listdir(directory)):
    if file.split(".")[-1] != "pkl":
        continue
    if file.split(".")[-2][-1] == "0":
        continue
    if file.split("type")[1][0] == "2" and file.split(".")[-2][-3] == "6":
        continue
    if file.split("type")[1][0] == "3" and file.split(".")[-2][-3] in ["5", "6"]:
        continue
    path = os.path.join(directory, file)
    with open(path, "rb") as f:
        data = pickle.load(f)

        dataset_used = file.split("type")[0].rstrip("_")

        all_datasets.append(dataset_used)
        all_type.append(file.split("type")[1][0])

        all_layers.append(data[0])
        all_qubits.append(data[1])
        all_num_params.append(data[2])
        all_gate_types.append(data[3])
        all_num_gates.append(data[4])
        all_depths.append(data[5])
        all_resources.append(data[6])
        all_original_parameters.append(data[7])
        all_trained_parameters.append(data[8])

        all_kta_values.append(data[9])

        all_metrics_before_training.append(data[10])
        all_metrics_before_testing.append(data[11])
        all_metrics_after_training.append((data[12]))
        all_metrics_after_testing.append(data[13])

    i = i + 1
    if i == files:
        break

df = pd.DataFrame(
    {
        "Layers": all_layers,
        "type": all_type,
        "num_qubits": all_qubits,
        "num_params": all_num_params,
        "gate_types": all_gate_types,
        "depths": all_depths,
        "num_gates": all_num_gates,
        "resources": all_resources,
        "org_para": all_original_parameters,
        "train_para": all_trained_parameters,
        "training_costs": all_kta_values,
        "before_scores_train": all_metrics_before_training,
        "before_scores_test": all_metrics_before_testing,
        "after_scores_train": all_metrics_after_training,
        "after_scores_test": all_metrics_after_testing,
        "data_used": all_datasets,
    }
)

df_boston = df[df.data_used == "boston_housing"]
df_exponential = df[df.data_used == "exponential"]
df_mod_x = df[df.data_used == "mod_x"]
df_polynomial = df[df.data_used == "polynomial"]
df_sine_wave = df[df.data_used == "sine_wave"]
df_UCI_air_quality = df[df.data_used == "UCI_air_quality"]
df_UCI_auto_mpg = df[df.data_used == "UCI_auto_mpg"]
df_UCI_automobile = df[df.data_used == "UCI_automobile"]
df_UCI_bike_share = df[df.data_used == "UCI_bike_share"]
df_UCI_computer_hardware = df[df.data_used == "UCI_computer_hardware"]
df_UCI_energy_efficiency = df[df.data_used == "UCI_energy_efficiency"]
df_UCI_forest_fires = df[df.data_used == "UCI_forest_fires"]
df_UCI_student_performance = df[df.data_used == "UCI_student_performance"]
df_UCI_wine_quality = df[df.data_used == "UCI_wine_quality"]

for data_used in [
    df_polynomial,
    df_sine_wave,
    df_exponential,
    df_mod_x,
    df_boston,
    df_UCI_air_quality,
    df_UCI_auto_mpg,
    df_UCI_automobile,
    df_UCI_bike_share,
    df_UCI_computer_hardware,
    df_UCI_energy_efficiency,
    df_UCI_forest_fires,
    df_UCI_student_performance,
    df_UCI_wine_quality,
]:
    scores_test_after = np.concatenate(
        data_used["after_scores_test"].to_numpy()
    ).reshape(58, 2)
    scores_test_before = np.concatenate(
        data_used["before_scores_test"].to_numpy()
    ).reshape(58, 2)

    index_rmse_test_after = np.argmin(scores_test_after[:, 0])
    index_mae_test_after = np.argmin(scores_test_after[:, 1])
    best_rmse_test_after = scores_test_after[index_rmse_test_after][0]
    best_mae_test_after = scores_test_after[index_mae_test_after][1]

    index_rmse_test_before = np.argmin(scores_test_before[:, 0])
    index_mae_test_before = np.argmin(scores_test_before[:, 1])
    best_rmse_test_before = scores_test_before[index_rmse_test_before][0]
    best_mae_test_before = scores_test_before[index_mae_test_before][1]

    all_best_scores_after_test.append(
        [
            index_rmse_test_after,
            best_rmse_test_after,
            index_mae_test_after,
            best_mae_test_after,
        ]
    )
    all_best_scores_before_test.append(
        [
            index_rmse_test_before,
            best_rmse_test_before,
            index_mae_test_before,
            best_mae_test_before,
        ]
    )

    data_l1 = data_used[data_used["type"] == "1"].sort_values(by="Layers")
    data_l2 = data_used[data_used["type"] == "2"]
    data_l3 = data_used[data_used["type"] == "3"]
    data_l4 = data_used[data_used["type"] == "4"].sort_values(by="Layers")
    data_l5 = data_used[data_used["type"] == "5"].sort_values(by="Layers")
    data_l6 = data_used[data_used["type"] == "6"].sort_values(by="Layers")
    data_l7 = data_used[data_used["type"] == "7"].sort_values(by="Layers")
    data_l8 = data_used[data_used["type"] == "8"].sort_values(by="Layers")

    data_table_rmse_after = []
    data_table_mae_after = []
    data_table_rmse_before = []
    data_table_mae_before = []

    plt.figure(figsize=(10, 5))
    for i, data_l in enumerate([data_l1, data_l4, data_l5, data_l6, data_l7, data_l8]):
        before_test_scores = np.concatenate(
            np.array(data_l["before_scores_test"])
        ).reshape(5, 2)
        after_test_scores = np.concatenate(
            np.array(data_l["after_scores_test"])
        ).reshape(5, 2)

        data_table_rmse_after.append(after_test_scores.min(axis=0)[0])
        data_table_mae_after.append(after_test_scores.min(axis=0)[1])
        data_table_rmse_before.append(before_test_scores.min(axis=0)[0])
        data_table_mae_before.append(before_test_scores.min(axis=0)[1])

        plt.plot(range(i * 5, 5 * (i + 1)), before_test_scores[:, 0], "r:.")
        # plt.plot(range(i * 5, 5 * (i + 1)), before_test_scores[:, 1], "b:.")
        plt.plot(range(i * 5, 5 * (i + 1)), after_test_scores[:, 0], "b:*")
        # plt.plot(range(i * 5, 5 * (i + 1)), after_test_scores[:, 1], "b:*")

    data_l2_2 = data_l2[data_l2["num_qubits"] == 2].sort_values(by="Layers")
    data_l2_4 = data_l2[data_l2["num_qubits"] == 4].sort_values(by="Layers")

    data_l3_1 = data_l3[data_l3["num_qubits"] == 1].sort_values(by="Layers")
    data_l3_2 = data_l3[data_l3["num_qubits"] == 2].sort_values(by="Layers")
    data_l3_3 = data_l3[data_l3["num_qubits"] == 3].sort_values(by="Layers")
    data_l3_4 = data_l3[data_l3["num_qubits"] == 4].sort_values(by="Layers")

    for i, data_l in enumerate([data_l2_2, data_l2_4]):
        before_test_scores = np.concatenate(
            np.array(data_l["before_scores_test"])
        ).reshape(4, 2)
        after_test_scores = np.concatenate(
            np.array(data_l["after_scores_test"])
        ).reshape(4, 2)

        data_table_rmse_after.append(after_test_scores.min(axis=0)[0])
        data_table_mae_after.append(after_test_scores.min(axis=0)[1])
        data_table_rmse_before.append(before_test_scores.min(axis=0)[0])
        data_table_mae_before.append(before_test_scores.min(axis=0)[1])

        plt.plot(range(30 + i * 4, 30 + 4 * (i + 1)), before_test_scores[:, 0], "r:.")
        # plt.plot(range(30 + i * 4, 30 + 4 * (i + 1)), before_test_scores[:, 1], "b:.")
        plt.plot(range(30 + i * 4, 30 + 4 * (i + 1)), after_test_scores[:, 0], "b:*")
        # plt.plot(range(30 + i * 4, 30 + 4 * (i + 1)), after_test_scores[:, 1], "b:*")

    for i, data_l in enumerate([data_l3_2, data_l3_2, data_l3_3, data_l3_4]):
        before_test_scores = np.concatenate(
            np.array(data_l["before_scores_test"])
        ).reshape(5, 2)
        after_test_scores = np.concatenate(
            np.array(data_l["after_scores_test"])
        ).reshape(5, 2)

        data_table_rmse_after.append(after_test_scores.min(axis=0)[0])
        data_table_mae_after.append(after_test_scores.min(axis=0)[1])
        data_table_rmse_before.append(before_test_scores.min(axis=0)[0])
        data_table_mae_before.append(before_test_scores.min(axis=0)[1])

        plt.plot(range(38 + i * 5, 38 + 5 * (i + 1)), before_test_scores[:, 0], "r:.")
        # plt.plot(range(38 + i * 5, 38 + 5 * (i + 1)), before_test_scores[:, 1], "b:.")
        plt.plot(range(38 + i * 5, 38 + 5 * (i + 1)), after_test_scores[:, 0], "b:*")
        # plt.plot(range(38 + i * 5, 38 + 5 * (i + 1)), after_test_scores[:, 1], "b:*")

    plt.xticks(
        list(range(58)),
        labels=[1, 2, 3, 4, 5] * 6 + [2, 3, 4, 5] * 2 + [1, 2, 3, 4, 5] * 4,
    )

    table_rmse_after.append(data_table_rmse_after)
    table_mae_after.append(data_table_mae_after)
    table_rmse_before.append(data_table_rmse_before)
    table_mae_before.append(data_table_mae_before)

    plt.legend(["Before KTA", "After KTA"], loc=2, framealpha=0.5)
    plt.ylabel("Validation RMSE")
    plt.xlabel("Number of Layers")

    xmin, xmax, ymin, ymax = plt.axis()

    plt.text(0 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 1")
    plt.text(5 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 4")
    plt.text(10 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 5")
    plt.text(15 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 6")
    plt.text(20 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 7")
    plt.text(25 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 8")
    plt.text(32 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 2")
    plt.text(46 - 0.5, ymin + 1.01 * (ymax - ymin), "Type 3")

    plt.text(
        29.5,
        ymin + 0.95 * (ymax - ymin),
        "n=2",
        bbox=dict(facecolor="red", alpha=0.4, boxstyle="round"),
        style="italic",
    )
    plt.text(
        29.5 + 4.5,
        ymin + 0.95 * (ymax - ymin),
        "n=4",
        bbox=dict(facecolor="red", alpha=0.4, boxstyle="round"),
        style="italic",
    )
    plt.text(
        29.5 + 9,
        ymin + 0.95 * (ymax - ymin),
        "n=1",
        bbox=dict(facecolor="red", alpha=0.4, boxstyle="round"),
        style="italic",
    )
    plt.text(
        29.5 + 13.5,
        ymin + 0.95 * (ymax - ymin),
        "n=2",
        bbox=dict(facecolor="red", alpha=0.4, boxstyle="round"),
        style="italic",
    )
    plt.text(
        29.5 + 19,
        ymin + 0.95 * (ymax - ymin),
        "n=3",
        bbox=dict(facecolor="red", alpha=0.4, boxstyle="round"),
        style="italic",
    )
    plt.text(
        29.5 + 23.5,
        ymin + 0.95 * (ymax - ymin),
        "n=4",
        bbox=dict(facecolor="red", alpha=0.4, boxstyle="round"),
        style="italic",
    )

    # plt.title(data_l.data_used.iloc[0])
    for x in range(4, 30, 5):
        plt.axvline(x, linestyle="--")
    plt.axvline(33, linestyle=":")
    plt.axvline(37, linestyle="--")
    plt.axvline(42, linestyle=":")
    plt.axvline(47, linestyle=":")
    plt.axvline(52, linestyle=":")
    plt.axvline(57, linestyle="--")
    plt.tight_layout()
    plt.savefig(f'new_data_/Regression/Plots/kernel_results_{data_l.data_used.iloc[0]}.pdf', format='pdf', dpi=1200,
                bbox_inches='tight')
    plt.show()

with open("new_data_/Regression/classical_results_regression.pkl", "rb") as f:
    classical_data = pickle.load(f)
classical_rmse = classical_data[0]
classical_mae = classical_data[1]

all_best_scores_after_test = np.asarray(all_best_scores_after_test)
all_best_scores_before_test = np.asarray(all_best_scores_before_test)

best_classical_rmse = np.min(classical_rmse, axis=1)
best_classical_mae = np.min(classical_mae, axis=1)

plt.plot(all_best_scores_after_test[:, 1], "c:+", label="After KTA")
plt.plot(all_best_scores_before_test[:, 1], "y:+", label="Before KTA")
# plt.plot(all_best_scores_after_test[:, 3], 'y:+', label='MAE')
plt.plot(best_classical_rmse, "r:.", label="Classical")
# plt.plot(best_classical_mae, 'b:.', label='Classical MAE')
plt.xlabel("Datasets")
plt.ylabel("Minimum Validation RMSE")
plt.xticks(
    range(14), ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
)
plt.legend()
plt.tight_layout()
# plt.savefig(f'new_data_/Regression/kernel_best_results.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

table_mae_after = np.array(table_mae_after)
table_rmse_after = np.array(table_rmse_after)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

table_rmse_after = pd.DataFrame(table_rmse_after,
                                columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                         'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])
table_mae_after = pd.DataFrame(table_mae_after,
                               columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                        'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                               index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])

print(table_rmse_after.round(3).min(axis=1))

table_rmse_after['Classical'] = best_classical_rmse
table_mae_after['Classical'] = best_classical_mae

print(table_rmse_after.round(3).min(axis=1))

table_mae_before = np.array(table_mae_before)
table_rmse_before = np.array(table_rmse_before)

table_rmse_before = pd.DataFrame(table_rmse_before,
                                 columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                          'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                 index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])
table_mae_before = pd.DataFrame(table_mae_before,
                                columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                         'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])

print(table_rmse_before.round(3).min(axis=1))

table_rmse_before['Classical'] = best_classical_rmse
table_mae_before['Classical'] = best_classical_mae

print(table_rmse_before.round(3).min(axis=1))
