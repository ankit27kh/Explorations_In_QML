import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from datasets import (
    circle,
    checkerboard,
    moons,
    UCI_wine_quality_classification,
    UCI_breast_cancer,
    UCI_bank,
    UCI_Adult,
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

all_acc = []
all_f1_weight = []
all_f1_macro = []
for dataset in [
    circle,
    checkerboard,
    moons,
    UCI_wine_quality_classification,
    UCI_breast_cancer,
    UCI_bank,
    UCI_Adult,
    concentric_circles,
    multiple_bands,
    four_circles,
    UCI_iris,
    MNIST,
    UCI_abalone,
    UCI_car,
    UCI_heart,
    UCI_wine,
]:

    X_train, X_test, y_train, y_test, dimension = dataset()

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_x_mm = MinMaxScaler([-1, 1])
    X_train_scaled = scaler_x_mm.fit_transform(X_train_scaled)
    X_test_scaled = scaler_x_mm.transform(X_test_scaled)

    dummy_1 = DummyClassifier(strategy="uniform", random_state=42)
    dummy_2 = DummyClassifier(strategy="stratified", random_state=42)
    dummy_3 = DummyClassifier(strategy="most_frequent")
    model_1 = LogisticRegression()
    model_2 = DecisionTreeClassifier(random_state=42)
    model_3 = SVC(random_state=42)
    reports = []
    acc = []
    weight_f1 = []
    macro_f1 = []
    for model in [dummy_1, dummy_2, dummy_3, model_1, model_2, model_3]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        reports.append(
            classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        )
        acc.append(reports[-1]["accuracy"])
        weight_f1.append(reports[-1]["weighted avg"]["f1-score"])
        macro_f1.append(reports[-1]["macro avg"]["f1-score"])
    print(acc)
    print(weight_f1)
    print(macro_f1)
    all_acc.append(acc)
    all_f1_macro.append(macro_f1)
    all_f1_weight.append(weight_f1)
"""
import pickle

save_data = np.asarray([all_acc, all_f1_weight, all_f1_macro])
with open('new_data_/Classification/classical_results_classification.pkl', 'wb') as f:
    pickle.dump(save_data, f)
"""
import seaborn as sns

sns.set_theme()
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 5))

all_acc = np.array(all_acc)
all_f1_weight = np.array(all_f1_weight)
all_f1_macro = np.array(all_f1_macro)
sns.lineplot(data=all_acc, ax=axes[0], marker='o', dashes=[(2, 2)] * 6, legend=False)
# sns.lineplot(data=all_f1_weight, ax=axes[1], marker='o', dashes=[(2, 2)] * 6, legend=False)
sns.lineplot(data=all_f1_macro, ax=axes[-1], marker='o', dashes=[(2, 2)] * 6, legend=False)
axes[0].set_title('Accuracy')
# axes[1].set_title('F1-Score (Weighted)')
axes[-1].set_title('F1-Score (Macro)')
# fig.supxlabel('Datasets')
fig.supylabel('Validation Scores')
plt.xticks(
    range(16),
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
)
plt.legend(['DC Uniform', 'DC Stratified', 'DC Most Frequent', 'Logistic Regression', 'Decision Tree', 'SVC'],
           bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.tight_layout()
# plt.savefig('new_data_/Classification/classical_class.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

all_acc = pd.DataFrame(all_acc)
