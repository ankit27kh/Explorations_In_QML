import itertools
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.program import UserMessenger

service = QiskitRuntimeService(instance='ibm-q-research-2/indian-inst-sci-1/main')
seed = 42
np.random.seed(seed)

num_points = 7

init_theta = np.linspace(0, np.pi, num_points)
init_phi = np.linspace(-np.pi, np.pi, num_points)

X = np.array(list(itertools.product(init_theta, init_phi)))

# Z-Hemispheres
y = np.array([-1 if i[0] <= np.pi / 2 else 1 for i in X])

# X-Hemispheres
# y = np.array([-1 if abs(i[1]) <= np.pi / 2 else 1 for i in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

meta = {
    "name": "kicked-top",
    "description": "A kicked-top classification program.",
    "max_execution_time": 1 * 60 * 60,  # 60 Minutes
    "spec": {},
}

print("Uploading program...")
program_id = service.upload_program(data="rotor_upload.py", metadata=meta)
print("Program Uploaded")

try:
    backend = service.backend("ibmq_qasm_simulator")
    # backend = service.backend("ibmq_lima")
    # backend = service.backend('ibmq_manila')
    options = {"backend_name": backend.name}

    inputs = {}
    inputs["X_train"] = X_train
    inputs["y_train"] = y_train
    inputs["X_test"] = X_test
    inputs["seed"] = seed
    inputs["layers"] = 5
    inputs["J"] = 1
    inputs["p"] = np.pi / 2
    inputs["steps"] = 10
    inputs["k"] = 10
    inputs["max_iter"] = 10
    inputs["shots"] = 1000
    # inputs['user_messenger'] = UserMessenger()
    weights = np.random.uniform(
        -np.pi, np.pi, [inputs["layers"], int(2 * inputs["J"]), 3]
    )
    weights = np.array(weights)
    inputs["params"] = weights

    print("Starting Job...")
    t = time.time()
    job = service.run(program_id, options=options, inputs=inputs)

    results = job.result()
    y_pred_train = results["Training Predictions"]
    y_pred_test = results["Testing Predictions"]

    print("Train Accuracy")
    print(accuracy_score(y_pred_train, y_train))
    print("Test Accuracy")
    print(accuracy_score(y_pred_test, y_test))
    print("Time Taken:", time.time() - t)
    print("Optimization Time on device", results["Optimisation Time"])
    print("Prediction Time", results["Prediction Time"])
except Exception as error:
    print(error)
finally:
    # Delete uploaded program
    print("Deleting program")
    service.delete_program(program_id)
    print("FINISH")
