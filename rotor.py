import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

sns.set_theme()

np.random.seed(42)

p = np.pi / 2
steps = 1000
num = 32


def get_X(x, y, z):
    return z * np.cos(k * x) + y * np.sin(k * x)


def get_Y(x, y, z):
    return y * np.cos(k * x) - z * np.sin(k * x)


def get_Z(x):
    return -x


def get_theta(z):
    return np.arccos(z)


def get_phi(x, y):
    return np.arctan2(y, x)


def init_x_y_z(theta, phi):
    return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)


init_theta = np.linspace(0, np.pi, num)
init_phi = np.linspace(-np.pi, np.pi, num)

all_acc_test = []

for k in [0.01, 0.1, 1, 2, 3, 4, 5, 6, 10, 50, 100]:
    data = []
    for th in init_theta:
        for ph in init_phi:
            x, y, z = init_x_y_z(th, ph)
            X, Y, Z = np.zeros(steps + 1), np.zeros(steps + 1), np.zeros(steps + 1)
            X[0] = x
            Y[0] = y
            Z[0] = z
            for i in range(1, steps + 1):
                X[i] = get_X(X[i - 1], Y[i - 1], Z[i - 1])
                Y[i] = get_Y(X[i - 1], Y[i - 1], Z[i - 1])
                Z[i] = get_Z(X[i - 1])

            theta, phi = [], []
            for x, y, z in zip(X, Y, Z):
                theta.append(get_theta(z))
                phi.append(get_phi(x, y))
            if th <= np.pi / 2:
                data.append([th, ph, theta[-1], phi[-1], 0])
            else:
                data.append([th, ph, theta[-1], phi[-1], 1])
            # plt.scatter(x=phi, y=theta)
            plt.scatter(x=phi, y=theta, rasterized=True)
    plt.xlabel(r"$\phi$")
    # plt.title(f"k={k}, steps={steps}")
    plt.ylabel(r"$\theta$", rotation=0)
    plt.xticks(
        [
            -np.pi,
            -3 * np.pi / 4,
            -np.pi / 2,
            -np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
        ],
        [
            r"$-\pi$",
            r"$-3\pi/4$",
            r"$-\pi/2$",
            r"$-\pi/4$",
            "$0$",
            r"$\pi/4$",
            r"$\pi/2$",
            r"$3\pi/4$",
            r"$\pi$",
        ],
    )
    plt.yticks(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
        ["$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
    )
    # plt.savefig(f"new_data_/rotor/classic/temp_{k}_{steps}.pdf", format="pdf", dpi=1200, bbox_inches="tight", )
    # plt.savefig(f'new_data_/rotor/classic/{k}_{steps}.eps', format='eps', dpi=1200, bbox_inches='tight')
    plt.show()

    data = np.array(data)

    X_data = data[:, 2:-1]
    y_data = data[:, -1]

    dummy_1 = DummyClassifier(strategy="uniform", random_state=42)
    dummy_2 = DummyClassifier(strategy="stratified", random_state=42)
    model_1 = LogisticRegression()
    model_2 = DecisionTreeClassifier(random_state=42)
    model_3 = SVC(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    reports = []
    acc = []
    scores = []
    for model in [dummy_1, dummy_2, model_1, model_2, model_3]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reports.append(
            classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        )
        acc.append(reports[-1]["accuracy"])
        scores.append(reports[-1]["weighted avg"]["f1-score"])
    print("For k = ", k)
    print(acc)
    all_acc_test.append(acc)
    # print(scores)

all_acc_test = np.array(all_acc_test).T

for acc_test in all_acc_test:
    plt.plot(acc_test, ":.")
plt.legend(
    [
        "Dummy - Uniform",
        "Dummy - Stratified",
        "Logistic Regression",
        "Decision Tree",
        "SVC",
    ]
)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.xticks(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.01, 0.1, 1, 2, 3, 4, 5, 6, 10, 50, 100]
)
# plt.savefig(f'new_data_/rotor/classic/classic_acc.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
