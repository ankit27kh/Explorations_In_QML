import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import seaborn as sns

sns.set_theme()

directory = "new_data_/rotor/noise"


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


i = 0
files = 36

# [use_noise, num_points, acc_train, acc_test]


all_accuracy_after_training = []
all_accuracy_after_testing = []
all_noise = []
all_num_points = []
all_case = []
for file in sorted_alphanumeric(os.listdir(directory)):
    if file.split(".")[-1] != "pkl":
        continue
    path = os.path.join(directory, file)
    with open(path, "rb") as f:
        data = pickle.load(f)
        legend = file.split("_")

        if legend[0] == "direct":
            all_case.append(1)
        else:
            all_case.append(2)
        all_accuracy_after_training.append(data[-2])
        all_accuracy_after_testing.append(data[-1])
        all_noise.append(data[0])
        all_num_points.append(data[1])

    i = i + 1
    if i == files:
        break

df = pd.DataFrame(
    {
        "case": all_case,
        "noise": all_noise,
        "points": all_num_points,
        "acc_train": all_accuracy_after_training,
        "acc_test": all_accuracy_after_testing,
    }
)

df_1_noise = df[df["case"] == 1][df["noise"] == True].sort_values(by="points")
df_1_no_noise = df[df["case"] == 1][df["noise"] == False].sort_values(by="points")
df_2_noise = df[df["case"] == 2][df["noise"] == True].sort_values(by="points")
df_2_no_noise = df[df["case"] == 2][df["noise"] == False].sort_values(by="points")

x = np.array([5, 6, 7, 8, 9, 10, 14, 17, 20])

for data in [df_1_no_noise, df_1_noise, df_2_noise]:
    scores = np.array(data["acc_test"]).T
    plt.plot(x, scores, ":.")
plt.ylabel("Validation Accuracy")
plt.xticks(x, x ** 2)
plt.xlabel("Total number of points in input data")
plt.legend(["Ideal", "Noisy Case 1", "Noisy Case 2"], loc="lower right")
plt.ylim(bottom=0.2)
# plt.title('Case 1')
# plt.savefig("new_data_/rotor/noise/case1.eps", format='eps', dpi=1200,bbox_inches='tight')
# plt.savefig("new_data_/rotor/noise/case_both.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

"""
for data in [df_2_no_noise, df_2_noise]:
    scores = np.array(data['acc_test']).T
    plt.plot(x, scores, ':.')
plt.ylabel("Validation Accuracy")
plt.xticks(x, x ** 2)
plt.xlabel("Total number of points in input data")
plt.legend(['Ideal', 'Noisy'], loc='lower right')
plt.ylim(bottom=0.2)
plt.title('Case 2')
# plt.savefig("new_data_/rotor/noise/case2.eps", format='eps', dpi=1200,bbox_inches='tight')
plt.show()
"""
