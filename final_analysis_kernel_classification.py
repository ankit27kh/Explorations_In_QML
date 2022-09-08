import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

sns.set_theme()

directory = "new_data_/Classification/Kernel"

"""
[num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters.numpy(),
 trained_parameters, scores, before_train_classification_report,
 before_test_classification_report, after_train_classification_report,
 after_test_classification_report]
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

all_classification_report_before_testing = []
all_classification_report_before_training = []
all_classification_report_after_training = []
all_classification_report_after_testing = []
all_accuracy_before_training = []
all_accuracy_before_testing = []
all_accuracy_after_training = []
all_accuracy_after_testing = []
all_weighted_f1_score_before_training = []
all_weighted_f1_score_before_testing = []
all_weighted_f1_score_after_training = []
all_weighted_f1_score_after_testing = []
all_macro_f1_score_before_training = []
all_macro_f1_score_before_testing = []
all_macro_f1_score_after_training = []
all_macro_f1_score_after_testing = []

all_ktas = []
all_best_scores_after_test = []
all_best_scores_before_test = []

i = 0
files = 58 * 16

table_acc_after = []
table_f1_macro_after = []
table_f1_weight_after = []
table_acc_before = []
table_f1_macro_before = []
table_f1_weight_before = []


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

        all_ktas.append(data[9])

        all_classification_report_before_training.append(data[10])
        all_classification_report_before_testing.append(data[11])
        all_classification_report_after_training.append((data[12]))
        all_classification_report_after_testing.append(data[13])

        all_accuracy_before_training.append(data[10]["accuracy"])
        all_accuracy_before_testing.append(data[11]["accuracy"])
        all_accuracy_after_training.append(data[12]["accuracy"])
        all_accuracy_after_testing.append(data[13]["accuracy"])
        all_weighted_f1_score_before_training.append(
            data[10]["weighted avg"]["f1-score"]
        )
        all_weighted_f1_score_before_testing.append(
            data[11]["weighted avg"]["f1-score"]
        )
        all_weighted_f1_score_after_training.append(
            data[12]["weighted avg"]["f1-score"]
        )
        all_weighted_f1_score_after_testing.append(data[13]["weighted avg"]["f1-score"])
        all_macro_f1_score_before_training.append(data[10]["macro avg"]["f1-score"])
        all_macro_f1_score_before_testing.append(data[11]["macro avg"]["f1-score"])
        all_macro_f1_score_after_training.append(data[12]["macro avg"]["f1-score"])
        all_macro_f1_score_after_testing.append(data[13]["macro avg"]["f1-score"])

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
        "training_costs": all_ktas,
        "before_accuracy_train": all_accuracy_before_training,
        "before_accuracy_test": all_accuracy_before_testing,
        "after_accuracy_train": all_accuracy_after_training,
        "after_accuracy_test": all_accuracy_after_testing,
        "before_weight_f1_train": all_weighted_f1_score_before_training,
        "before_weight_f1_test": all_weighted_f1_score_before_testing,
        "after_weight_f1_train": all_weighted_f1_score_after_training,
        "after_weight_f1_test": all_weighted_f1_score_after_testing,
        "before_macro_f1_train": all_macro_f1_score_before_training,
        "before_macro_f1_test": all_macro_f1_score_before_testing,
        "after_macro_f1_train": all_macro_f1_score_after_training,
        "after_macro_f1_test": all_macro_f1_score_after_testing,
        "data_used": all_datasets,
    }
)

df_checkerboard = df[df.data_used == "checkerboard"]
df_circle = df[df.data_used == "circle"]
df_moons = df[df.data_used == "moons"]
df_multiple_bands = df[df.data_used == "multiple_bands"]
df_UCI_iris = df[df.data_used == "UCI_iris"]
df_MNIST = df[df.data_used == "MNIST"]
df_UCI_abalone = df[df.data_used == "UCI_abalone"]
df_UCI_car = df[df.data_used == "UCI_car"]
df_UCI_heart = df[df.data_used == "UCI_heart"]
df_UCI_wine = df[df.data_used == "UCI_wine"]
df_UCI_wine_quality_classification = df[
    df.data_used == "UCI_wine_quality_classification"
    ]
df_UCI_breast_cancer = df[df.data_used == "UCI_breast_cancer"]
df_UCI_bank = df[df.data_used == "UCI_bank"]
df_UCI_Adult = df[df.data_used == "UCI_Adult"]
df_four_circles = df[df.data_used == "four_circles"]
df_concentric_circles = df[df.data_used == "concentric_circles"]

for data_used in [
    df_circle,
    df_checkerboard,
    df_moons,
    df_UCI_wine_quality_classification,
    df_UCI_breast_cancer,
    df_UCI_bank,
    df_UCI_Adult,
    df_concentric_circles,
    df_multiple_bands,
    df_four_circles,
    df_UCI_iris,
    df_MNIST,
    df_UCI_abalone,
    df_UCI_car,
    df_UCI_heart,
    df_UCI_wine,
]:

    acc_test_after = data_used["after_accuracy_test"].to_numpy()
    macro_test_f1_after = data_used["after_macro_f1_test"].to_numpy()
    weight_test_f1_after = data_used["after_weight_f1_test"].to_numpy()

    acc_test_before = data_used["before_accuracy_test"].to_numpy()
    macro_test_f1_before = data_used["before_macro_f1_test"].to_numpy()
    weight_test_f1_before = data_used["before_weight_f1_test"].to_numpy()

    index_acc = np.argmax(acc_test_after)
    best_acc = acc_test_after[index_acc]

    index_f1_macro = np.argmax(macro_test_f1_after)
    best_f1_macro = macro_test_f1_after[index_f1_macro]

    index_f1_weight = np.argmax(weight_test_f1_after)
    best_f1_weight = weight_test_f1_after[index_f1_weight]

    index_acc_before = np.argmax(acc_test_before)
    best_acc_before = acc_test_before[index_acc_before]

    index_f1_macro_before = np.argmax(macro_test_f1_before)
    best_f1_macro_before = macro_test_f1_before[index_f1_macro_before]

    index_f1_weight_before = np.argmax(weight_test_f1_before)
    best_f1_weight_before = weight_test_f1_before[index_f1_weight_before]

    all_best_scores_after_test.append(
        [
            index_acc,
            best_acc,
            index_f1_macro,
            best_f1_macro,
            index_f1_weight,
            best_f1_weight,
        ]
    )

    all_best_scores_before_test.append(
        [
            index_acc_before,
            best_acc_before,
            index_f1_macro_before,
            best_f1_macro_before,
            index_f1_weight_before,
            best_f1_weight_before,
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

    data_table_acc_after = []
    data_table_f1_macro_after = []
    data_table_f1_weight_after = []
    data_table_acc_before = []
    data_table_f1_macro_before = []
    data_table_f1_weight_before = []

    plt.figure(figsize=(10, 5))
    for i, data_l in enumerate([data_l1, data_l4, data_l5, data_l6, data_l7, data_l8]):
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.after_accuracy_test, "r:.")
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.after_weight_f1_test, "b:.")
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.after_macro_f1_test, "c:.")
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.before_accuracy_test, "m:+")
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.before_weight_f1_test, "y:+")
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.before_macro_f1_test, "g:+")

        data_table_acc_after.append(data_l.after_accuracy_test.max())
        data_table_f1_macro_after.append(data_l.after_macro_f1_test.max())
        data_table_f1_weight_after.append(data_l.after_weight_f1_test.max())

        data_table_acc_before.append(data_l.before_accuracy_test.max())
        data_table_f1_macro_before.append(data_l.before_macro_f1_test.max())
        data_table_f1_weight_before.append(data_l.before_weight_f1_test.max())

    data_l2_2 = data_l2[data_l2["num_qubits"] == 2].sort_values(by="Layers")
    data_l2_4 = data_l2[data_l2["num_qubits"] == 4].sort_values(by="Layers")

    data_l3_1 = data_l3[data_l3["num_qubits"] == 1].sort_values(by="Layers")
    data_l3_2 = data_l3[data_l3["num_qubits"] == 2].sort_values(by="Layers")
    data_l3_3 = data_l3[data_l3["num_qubits"] == 3].sort_values(by="Layers")
    data_l3_4 = data_l3[data_l3["num_qubits"] == 4].sort_values(by="Layers")

    plt.plot(range(30, 34), data_l2_2.after_accuracy_test, "r:.")
    plt.plot(range(30, 34), data_l2_2.after_weight_f1_test, "b:.")
    plt.plot(range(30, 34), data_l2_2.after_macro_f1_test, "c:.")

    plt.plot(range(34, 38), data_l2_4.after_accuracy_test, "r:.")
    plt.plot(range(34, 38), data_l2_4.after_weight_f1_test, "b:.")
    plt.plot(range(34, 38), data_l2_4.after_macro_f1_test, "c:.")

    plt.plot(range(30, 34), data_l2_2.before_accuracy_test, "r:*")
    plt.plot(range(30, 34), data_l2_2.before_weight_f1_test, "b:*")
    plt.plot(range(30, 34), data_l2_2.before_macro_f1_test, "c:*")

    plt.plot(range(34, 38), data_l2_4.before_accuracy_test, "r:*")
    plt.plot(range(34, 38), data_l2_4.before_weight_f1_test, "b:*")
    plt.plot(range(34, 38), data_l2_4.before_macro_f1_test, "c:*")

    data_table_acc_after.append(data_l2_2.after_accuracy_test.max())
    data_table_f1_macro_after.append(data_l2_2.after_macro_f1_test.max())
    data_table_f1_weight_after.append(data_l2_2.after_weight_f1_test.max())
    data_table_acc_after.append(data_l2_4.after_accuracy_test.max())
    data_table_f1_macro_after.append(data_l2_4.after_macro_f1_test.max())
    data_table_f1_weight_after.append(data_l2_4.after_weight_f1_test.max())

    data_table_acc_before.append(data_l2_2.before_accuracy_test.max())
    data_table_f1_macro_before.append(data_l2_2.before_macro_f1_test.max())
    data_table_f1_weight_before.append(data_l2_2.before_weight_f1_test.max())
    data_table_acc_before.append(data_l2_4.before_accuracy_test.max())
    data_table_f1_macro_before.append(data_l2_4.before_macro_f1_test.max())
    data_table_f1_weight_before.append(data_l2_4.before_weight_f1_test.max())

    for i, data_l3_l in enumerate([data_l3_1, data_l3_2, data_l3_3, data_l3_4]):
        plt.plot(
            range(38 + i * 5, 38 + 5 * (i + 1)), data_l3_l.after_accuracy_test, "r:."
        )
        plt.plot(
            range(38 + i * 5, 38 + 5 * (i + 1)), data_l3_l.after_weight_f1_test, "b:."
        )
        plt.plot(
            range(38 + i * 5, 38 + 5 * (i + 1)), data_l3_l.after_macro_f1_test, "c:."
        )

        plt.plot(
            range(38 + i * 5, 38 + 5 * (i + 1)), data_l3_l.before_accuracy_test, "m:+"
        )
        plt.plot(
            range(38 + i * 5, 38 + 5 * (i + 1)), data_l3_l.before_weight_f1_test, "y:+"
        )
        plt.plot(
            range(38 + i * 5, 38 + 5 * (i + 1)), data_l3_l.before_macro_f1_test, "g:+"
        )

        data_table_acc_after.append(data_l3_l.after_accuracy_test.max())
        data_table_f1_macro_after.append(data_l3_l.after_macro_f1_test.max())
        data_table_f1_weight_after.append(data_l3_l.after_weight_f1_test.max())

        data_table_acc_before.append(data_l3_l.before_accuracy_test.max())
        data_table_f1_macro_before.append(data_l3_l.before_macro_f1_test.max())
        data_table_f1_weight_before.append(data_l3_l.before_weight_f1_test.max())

    plt.xticks(
        list(range(58)),
        labels=[1, 2, 3, 4, 5] * 6 + [2, 3, 4, 5] * 2 + [1, 2, 3, 4, 5] * 4,
    )

    plt.legend(["Accuracy After", "F1-Weighted After", "F1-Macro After", "Accuracy Before", "F1-Weighted Before",
                "F1-Macro Before"], loc=3, framealpha=.5)
    plt.ylabel("Validation Score")
    plt.xlabel("Number of Layers")
    # plt.title(data_l.data_used.iloc[0])
    for x in range(4, 30, 5):
        plt.axvline(x, linestyle="--")
    plt.axvline(33, linestyle=":")
    plt.axvline(37, linestyle="--")
    plt.axvline(42, linestyle=":")
    plt.axvline(47, linestyle=":")
    plt.axvline(52, linestyle=":")
    plt.axvline(57, linestyle="--")

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

    table_acc_after.append(data_table_acc_after)
    table_f1_macro_after.append(data_table_f1_macro_after)
    table_f1_weight_after.append(data_table_f1_weight_after)

    table_acc_before.append(data_table_acc_before)
    table_f1_macro_before.append(data_table_f1_macro_before)
    table_f1_weight_before.append(data_table_f1_weight_before)

    plt.tight_layout()
    # plt.savefig(f'new_data_/Classification/Plots/kernel_results_{data_used.data_used.iloc[0]}.pdf', format='pdf',
    #            dpi=1200, bbox_inches='tight')
    plt.show()

with open("new_data_/Classification/classical_results_classification.pkl", "rb") as f:
    classical_data = pickle.load(f)
classical_acc = classical_data[0]
classical_f1_weight = classical_data[1]
classical_f1_macro = classical_data[2]

all_best_scores_after_test = np.asarray(all_best_scores_after_test)
all_best_scores_before_test = np.asarray(all_best_scores_before_test)

best_classical_acc = np.max(classical_acc, axis=1)
best_classical_f1_weight = np.max(classical_f1_weight, axis=1)
best_classical_f1_macro = np.max(classical_f1_macro, axis=1)

plt.plot(all_best_scores_after_test[:, 1], ":+", label="After KTA")
plt.plot(all_best_scores_before_test[:, 1], ":*", label="Before KTA")
plt.plot(best_classical_acc, ":.", label="Classical")
plt.xlabel("Datasets")
plt.ylabel("Maximum Validation Accuracy")
plt.xticks(
    range(16),
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
)
plt.legend(loc=3, framealpha=.5)
plt.tight_layout()
# plt.savefig(f'new_data_/Classification/class_kernel_best_acc.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

plt.plot(all_best_scores_after_test[:, 3], ":+", label="After KTA")
plt.plot(all_best_scores_before_test[:, 3], ":*", label="Before KTA")
plt.plot(best_classical_f1_macro, ":.", label="Classical")
plt.xlabel("Datasets")
plt.ylabel("Maximum Validation F1-Score (Macro)")
plt.xticks(
    range(16),
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
)
plt.legend(loc=3, framealpha=.5)
plt.tight_layout()
# plt.savefig(f'new_data_/Classification/class_kernel_best_macro_f1.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

plt.plot(all_best_scores_after_test[:, 5], ":+", label="After KTA")
plt.plot(all_best_scores_before_test[:, 5], ":*", label="Before KTA")
plt.plot(best_classical_f1_weight, ":.", label="Classical")
plt.xlabel("Datasets")
plt.ylabel("Maximum Validation F1-Score (Weighted)")
plt.xticks(
    range(16),
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
)
plt.legend(loc=3, framealpha=.5)
plt.tight_layout()
# plt.savefig(f'new_data_/Classification/class_kernel_best_weight_f1.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

table_f1_macro_after = np.array(table_f1_macro_after)
table_f1_weight_after = np.array(table_f1_weight_after)
table_acc_after = np.array(table_acc_after)

table_f1_macro_after = pd.DataFrame(table_f1_macro_after,
                                    columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                             'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                    index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O',
                                           'P'])
table_f1_weight_after = pd.DataFrame(table_f1_weight_after,
                                     columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                              'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                     index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O',
                                            'P'])
table_acc_after = pd.DataFrame(table_acc_after,
                               columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                        'Type 2-4',
                                        'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                               index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O', 'P'])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(table_acc_after.round(3).max(axis=1))

table_acc_after['Classical'] = best_classical_acc
table_f1_weight_after['Classical'] = best_classical_f1_weight
table_f1_macro_after['Classical'] = best_classical_f1_macro

print(table_acc_after.round(3).max(axis=1))

table_f1_macro_before = np.array(table_f1_macro_before)
table_f1_weight_before = np.array(table_f1_weight_before)
table_acc_before = np.array(table_acc_before)

table_f1_macro_before = pd.DataFrame(table_f1_macro_before,
                                     columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                              'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                     index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O',
                                            'P'])
table_f1_weight_before = pd.DataFrame(table_f1_weight_before,
                                      columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                               'Type 2-4', 'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                      index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O',
                                             'P'])
table_acc_before = pd.DataFrame(table_acc_before,
                                columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 2-2',
                                         'Type 2-4',
                                         'Type 3-1', 'Type 3-2', 'Type 3-3', 'Type 3-4'],
                                index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O', 'P'])

print(table_acc_before.round(3).max(axis=1))

table_acc_before['Classical'] = best_classical_acc
table_f1_weight_before['Classical'] = best_classical_f1_weight
table_f1_macro_before['Classical'] = best_classical_f1_macro

print(table_acc_before.round(3).max(axis=1))
