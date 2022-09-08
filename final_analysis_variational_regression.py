import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

sns.set_theme()

directory = "new_data_/Regression/Variational"

"""
[num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters.numpy(),
     trained_parameters, training_costs, rmse_mae_training, rmse_mae_testing]
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
all_training_costs = []
all_rmse_mae_train = []
all_rmse_mae_test = []
all_type = []
all_datasets = []
all_best_scores_after_test = []
all_best_scores_after_train = []

i = 0
files = 36 * 14

table_rmse = []
table_mae = []


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


for file in sorted_alphanumeric(os.listdir(directory)):
    if file.split(".")[-1] != "pkl":
        continue
    if file.split(".")[-2][-1] == "0":
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

        all_training_costs.append(data[9])
        all_rmse_mae_train.append(np.asarray(data[10]))
        all_rmse_mae_test.append(np.asarray(data[11]))

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
        "training_costs": all_training_costs,
        "scores_train": all_rmse_mae_train,
        "scores_test": all_rmse_mae_test,
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
    scores_test_after = np.concatenate(data_used["scores_test"].to_numpy()).reshape(
        30, 2, 2
    )[:, 1, :]
    scores_train_after = np.concatenate(data_used["scores_train"].to_numpy()).reshape(
        30, 2, 2
    )[:, 1, :]

    index_rmse_test = np.argmin(scores_test_after[:, 0])
    index_mae_test = np.argmin(scores_test_after[:, 1])
    best_rmse_test = scores_test_after[index_rmse_test][0]
    best_mae_test = scores_test_after[index_mae_test][1]

    index_rmse_train = np.argmin(scores_train_after[:, 0])
    index_mae_train = np.argmin(scores_train_after[:, 1])
    best_rmse_train = scores_train_after[index_rmse_train][0]
    best_mae_train = scores_train_after[index_mae_train][1]

    all_best_scores_after_test.append(
        [index_rmse_test, best_rmse_test, index_mae_test, best_mae_test]
    )
    all_best_scores_after_train.append(
        [index_rmse_train, best_rmse_train, index_mae_train, best_mae_train]
    )

    data_l1 = data_used[data_used["type"] == "1"].sort_values(by="Layers")
    data_l4 = data_used[data_used["type"] == "4"].sort_values(by="Layers")
    data_l5 = data_used[data_used["type"] == "5"].sort_values(by="Layers")
    data_l6 = data_used[data_used["type"] == "6"].sort_values(by="Layers")
    data_l7 = data_used[data_used["type"] == "7"].sort_values(by="Layers")
    data_l8 = data_used[data_used["type"] == "8"].sort_values(by="Layers")

    data_table_rmse = []
    data_table_mae = []

    plt.figure(figsize=(8, 5))
    for i, data_l in enumerate([data_l1, data_l4, data_l5, data_l6, data_l7, data_l8]):
        after_training_test_scores = np.concatenate(
            np.array(data_l["scores_test"])
        ).reshape(5, 2, 2)[:, 1]
        after_training_train_scores = np.concatenate(
            np.array(data_l["scores_train"])
        ).reshape(5, 2, 2)[:, 1]

        data_table_rmse.append(after_training_test_scores.min(axis=0)[0])
        data_table_mae.append(after_training_test_scores.min(axis=0)[1])

        # before_training_test_scores = np.concatenate(np.array(data_l['scores_test'])).reshape(5, 2, 2)[:, 0]

        plt.plot(range(i * 5, 5 * (i + 1)), after_training_test_scores[:, 0], "r:.")
        plt.plot(range(i * 5, 5 * (i + 1)), after_training_test_scores[:, 1], "b:.")

        # plt.plot(range(i * 6, 6 * (i + 1)), before_training_test_scores, ':.')
    plt.xticks(list(range(30)), labels=[1, 2, 3, 4, 5] * 6)

    table_rmse.append(data_table_rmse)
    table_mae.append(data_table_mae)

    plt.legend(
        [
            "RMSE",
            "MAE",
        ],
        loc=1,
        framealpha=0.5,
    )
    # 'RMSE before training', 'MAE before training'])
    plt.ylabel("Validation Error")
    plt.xlabel("Number of Layers")
    # plt.title(data_l.data_used.iloc[0])

    xmin, xmax, ymin, ymax = plt.axis()

    plt.text(0.5, ymin + 1.01 * (ymax - ymin), "Type 1")
    plt.text(5.5, ymin + 1.01 * (ymax - ymin), "Type 4")
    plt.text(10.5, ymin + 1.01 * (ymax - ymin), "Type 5")
    plt.text(15.5, ymin + 1.01 * (ymax - ymin), "Type 6")
    plt.text(20.5, ymin + 1.01 * (ymax - ymin), "Type 7")
    plt.text(25.5, ymin + 1.01 * (ymax - ymin), "Type 8")
    for x in range(4, 30, 5):
        plt.axvline(x, linestyle=":")
    plt.tight_layout()
    # plt.savefig(f'new_data_/Regression/Plots/variational_results_{data_l.data_used.iloc[0]}.pdf', format='pdf',dpi=1200, bbox_inches='tight')
    plt.show()

with open("new_data_/Regression/classical_results_regression.pkl", "rb") as f:
    classical_data = pickle.load(f)
classical_rmse = classical_data[0]
classical_mae = classical_data[1]

all_best_scores_after_test = np.asarray(all_best_scores_after_test)
all_best_scores_after_train = np.asarray(all_best_scores_after_train)

best_classical_rmse = np.min(classical_rmse, axis=1)
best_classical_mae = np.min(classical_mae, axis=1)

plt.plot(all_best_scores_after_test[:, 1], "c:+", label="RMSE")
plt.plot(all_best_scores_after_test[:, 3], "y:+", label="MAE")
plt.plot(best_classical_rmse, "r:.", label="Classical RMSE")
plt.plot(best_classical_mae, "b:.", label="Classical MAE")
plt.xlabel("Datasets")
plt.ylabel("Minimum Validation Error")
plt.xticks(
    range(14), ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
)
plt.legend()
plt.tight_layout()
# plt.savefig(f'new_data_/Regression/variational_best_results.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

table_mae = np.array(table_mae)
table_rmse = np.array(table_rmse)

table_rmse = pd.DataFrame(table_rmse, columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8'],
                          index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])
table_mae = pd.DataFrame(table_mae, columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8'],
                         index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])

table_rmse['Classical'] = best_classical_rmse
table_mae['Classical'] = best_classical_mae
