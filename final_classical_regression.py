import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from datasets import (
    polynomial,
    sine_wave,
    exponential,
    mod_x,
    boston_housing,
    UCI_air_quality,
    UCI_auto_mpg,
    UCI_automobile,
    UCI_bike_share,
    UCI_computer_hardware,
    UCI_energy_efficiency,
    UCI_forest_fires,
    UCI_student_performance,
    UCI_wine_quality,
)

all_rmse = []
all_mae = []
for dataset in [
    polynomial,
    sine_wave,
    exponential,
    mod_x,
    boston_housing,
    UCI_air_quality,
    UCI_auto_mpg,
    UCI_automobile,
    UCI_bike_share,
    UCI_computer_hardware,
    UCI_energy_efficiency,
    UCI_forest_fires,
    UCI_student_performance,
    UCI_wine_quality,
]:

    X_train, X_test, y_train, y_test, dimension = dataset()

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    scaler_x_mm = MinMaxScaler([-1, 1])
    X_train_scaled = scaler_x_mm.fit_transform(X_train_scaled)
    X_test_scaled = scaler_x_mm.transform(X_test_scaled)

    scaler_y_mm = MinMaxScaler([-1, 1])
    y_train_scaled = scaler_y_mm.fit_transform(y_train_scaled)
    y_test_scaled = scaler_y_mm.transform(y_test_scaled)

    dummy_1 = DummyRegressor(strategy="mean")
    dummy_2 = DummyRegressor(strategy="median")
    model_1 = LinearRegression()
    model_2 = DecisionTreeRegressor(random_state=42)
    model_3 = SVR()
    rmse = []
    mae = []
    for model in [dummy_1, dummy_2, model_1, model_2, model_3]:
        model.fit(X_train_scaled, y_train_scaled.ravel())
        y_pred = model.predict(X_test_scaled)
        rmse.append(np.sqrt(mean_squared_error(y_test_scaled, y_pred)))
        mae.append(mean_absolute_error(y_test_scaled, y_pred))

    print(rmse)
    print(mae)
    all_rmse.append(rmse)
    all_mae.append(mae)
"""
import pickle

save_data = np.asarray([all_rmse, all_mae])
with open('new_data_/Regression/classical_results_regression.pkl', 'wb') as f:
    pickle.dump(save_data, f)
"""
import seaborn as sns

sns.set_theme()
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 5))

all_rmse = np.array(all_rmse)
all_mae = np.array(all_mae)
sns.lineplot(data=all_rmse, ax=axes[0], marker='o', dashes=[(2, 2)] * 5, legend=False)
sns.lineplot(data=all_mae, ax=axes[-1], marker='o', dashes=[(2, 2)] * 5, legend=False)
axes[0].set_title('RMSE')
# axes[1].set_title('F1-Score (Weighted)')
axes[-1].set_title('MAE')
# fig.supxlabel('Datasets')
fig.supylabel('Validation Error')
plt.xticks(
    range(14),
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"],
)
plt.legend(['DR Mean', 'DR Median', 'Linear Regression', 'Decision Tree', 'SVR'],
           bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.tight_layout()
# plt.savefig('new_data_/Regression/classical_regression.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

all_rmse = pd.DataFrame(all_rmse)
