import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

sns.set_theme()

directory = "new_data_/Classification/Variational"

"""
[num_layers, num_qubits, num_params, gate_types, total_gates, depth, resources, original_parameters.numpy(),
     trained_parameters, scores, before_train_classification_report, before_test_classification_report,
     after_train_classification_report, after_test_classification_report]
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

all_log_loss = []
all_best_scores_after_test = []

i = 0
files = 528

table_acc = []
table_f1_macro = []
table_f1_weight = []


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

        all_log_loss.append(data[9])

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
        "training_costs": all_log_loss,
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

    index_acc = np.argmax(acc_test_after)
    best_acc = acc_test_after[index_acc]

    index_f1_macro = np.argmax(macro_test_f1_after)
    best_f1_macro = macro_test_f1_after[index_f1_macro]

    index_f1_weight = np.argmax(weight_test_f1_after)
    best_f1_weight = weight_test_f1_after[index_f1_weight]

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

    data_l1 = data_used[data_used["type"] == "1"].sort_values(by="Layers")
    data_l4 = data_used[data_used["type"] == "4"].sort_values(by="Layers")
    data_l5 = data_used[data_used["type"] == "5"].sort_values(by="Layers")
    data_l6 = data_used[data_used["type"] == "6"].sort_values(by="Layers")
    data_l7 = data_used[data_used["type"] == "7"].sort_values(by="Layers")
    data_l8 = data_used[data_used["type"] == "8"].sort_values(by="Layers")

    data_table_acc = []
    data_table_f1_macro = []
    data_table_f1_weight = []

    plt.figure(figsize=(8, 5))
    for i, data_l in enumerate([data_l1, data_l4, data_l5, data_l6, data_l7, data_l8]):
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.after_accuracy_test, "r:.")
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.after_weight_f1_test, "b:.")
        plt.plot(range(i * 5, 5 * (i + 1)), data_l.after_macro_f1_test, "c:.")

        data_table_acc.append(data_l.after_accuracy_test.max())
        data_table_f1_macro.append(data_l.after_macro_f1_test.max())
        data_table_f1_weight.append(data_l.after_weight_f1_test.max())

    plt.xticks(list(range(30)), labels=[1, 2, 3, 4, 5] * 6)
    plt.legend(["Accuracy", "F1-Weighted", "F1-Macro"], loc=3, framealpha=0.5)
    plt.ylabel("Validation Score")
    plt.xlabel("Number of Layers")
    # plt.title(data_used.data_used.iloc[0])

    plt.tight_layout()

    for x in range(4, 30, 5):
        plt.axvline(x, linestyle=":")

    table_acc.append(data_table_acc)
    table_f1_macro.append(data_table_f1_macro)
    table_f1_weight.append(data_table_f1_weight)

    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(0.5, ymin + 1.01 * (ymax - ymin), "Type 1")
    plt.text(5.5, ymin + 1.01 * (ymax - ymin), "Type 4")
    plt.text(10.5, ymin + 1.01 * (ymax - ymin), "Type 5")
    plt.text(15.5, ymin + 1.01 * (ymax - ymin), "Type 6")
    plt.text(20.5, ymin + 1.01 * (ymax - ymin), "Type 7")
    plt.text(25.5, ymin + 1.01 * (ymax - ymin), "Type 8")

    # plt.savefig(f'new_data_/Classification/Plots/variational_results_{data_used.data_used.iloc[0]}.pdf', format='pdf',
    #            dpi=1200, bbox_inches='tight')
    plt.show()

with open("new_data_/Classification/classical_results_classification.pkl", "rb") as f:
    classical_data = pickle.load(f)
classical_acc = classical_data[0]
classical_f1_weight = classical_data[1]
classical_f1_macro = classical_data[2]
all_best_scores_after_test = np.asarray(all_best_scores_after_test)

best_classical_acc = np.max(classical_acc, axis=1)
best_classical_f1_weight = np.max(classical_f1_weight, axis=1)
best_classical_f1_macro = np.max(classical_f1_macro, axis=1)

plt.plot(all_best_scores_after_test[:, 1], "r:+", label="Accuracy")
plt.plot(all_best_scores_after_test[:, 3], "b:+", label="F1-Macro")
plt.plot(all_best_scores_after_test[:, 5], "c:+", label="F1-Weighted")

plt.plot(best_classical_acc, "m:.", label="Classical Accuracy")
plt.plot(best_classical_f1_weight, "y:.", label="Classical F1-Weight")
plt.plot(best_classical_f1_macro, "g:.", label="Classical F1-Macro")

plt.xlabel("Datasets")
plt.ylabel("Maximum Validation Score")
plt.xticks(
    range(16),
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
)
plt.legend()
plt.tight_layout()
# plt.savefig(f'new_data_/Classification/class_variational_best_results.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

table_f1_macro = np.array(table_f1_macro)
table_f1_weight = np.array(table_f1_weight)
table_acc = np.array(table_acc)

table_f1_macro = pd.DataFrame(table_f1_macro, columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8'],
                              index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O', 'P'])
table_f1_weight = pd.DataFrame(table_f1_weight, columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8'],
                               index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O', 'P'])
table_acc = pd.DataFrame(table_acc, columns=['Type 1', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8'],
                         index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 'O', 'P'])

table_acc['Classical'] = best_classical_acc
table_f1_weight['Classical'] = best_classical_f1_weight
table_f1_macro['Classical'] = best_classical_f1_macro
