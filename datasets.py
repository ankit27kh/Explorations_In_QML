"""
This file contains the different datasets used for comparing the models.
Artificial datasets:
1. Circle -> Binary Classification (2D)
2. Concentric Circles -> Multiclass Classification (2D)
3. Checkerboard -> Binary Classification (2D)
5. Multiple Bands -> Multiclass Classification (2D)
6. Four Circles -> Multiclass Classification (2D)
6. Polynomial -> Regression (1D)
7. Sine wave -> Regression (1D)
8. Blobs -> Multiclass Classification (nD)
9. Moons -> Binary Classification (2D)
10. Random Regression -> Regression (nD)
11. Exponential -> Regression (1D)
12. Mod x -> Regression (1D)
Real-World datasets:
1. Iris -> Multiclass Classification
2. Boston Housing -> Regression
3. Air Quality -> Regression
4. Auto mpg -> Regression
5. Automobile -> Regression
6. Bike Share -> Regression
7. Computer Hardware -> Regression
8. Energy Efficiency -> Regression
9. Forest Fires-> Regression
10. Student Performance-> Regression
11. Wine Quality-> Regression/Binary Classification
12. MNIST Digits -> Multiclass Classification
13. Abalone -> Multiclass Classification
14. Breast Cancer -> Binary Classification
15. Car -> Multiclass Classification
16. Heart Disease -> Multiclass Classification
17. Wine -> Multiclass Classification
18. Bank -> Binary Classification
19. Adult -> Binary Classification
"""
import pandas as pd
import pennylane.numpy as np
from sklearn import datasets as dt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 42
np.random.seed(seed)


def circle(points=1000, test_size=0.3):
    """
    Create binary distinguishable data with 50% points inside the circle (0) and 50% points outside the circle (1).
    This implies choosing a simple classifier which sets all to 0 or 1 will be 50% accurate.
    Our classifier must improve on this.

    :param points:
    Number of points in the dataset. Default = 1000
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    return concentric_circles(points, test_size, classes=2)


def concentric_circles(points=1000, test_size=0.3, classes=3):
    """
    Create multiclass classification data. Creates 'classes' - 1 concentric circles. Classes goes from 0 to 'classes-1'
    from inside to outside.

    :param points:
    Number of points in the dataset. Default = 1000
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :param classes:
    Number of classes. Default = 3
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    if points < 25 * classes:
        print("Too few points")
        print(f"Setting points to {25 * classes}")
        points = 25 * classes
    radii = np.ones([classes - 1])
    for i in range(1, classes):
        radii[i - 1] = np.sqrt(i * 4 / classes / np.pi)
    X = (np.random.random([points, 2]) * 2) - 1
    y = (
            np.ones(
                [
                    points,
                ]
            )
            * classes
    )
    for i in range(1, classes):
        for j, p in enumerate(X):
            if np.linalg.norm(p) <= radii[-i]:
                y[j] = classes - i
    y = [int(i) - 1 for i in y]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        2,
    )


def multiple_bands(per_points=100, test_size=0.3, classes=3):
    """
    Create multiclass classification data. Creates 'classes' - 1 parallel bands. Classes goes from 0 to 'classes-1'
    from left to right.

    :param per_points:
    Number of points in each band. Default = 100
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :param classes:
    Number of classes. Default = 3
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    centres = np.linspace(-classes, classes, classes)
    X = []
    y = []
    for i in range(classes):
        x1 = np.random.uniform(centres[i] - 1, centres[i] + 1, per_points)
        x2 = np.random.uniform(-1, 1, per_points)
        x = [[i, j] for i, j in zip(x1, x2)]
        X.extend(x)
        y.extend(np.ones(per_points, dtype=int) * i)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        2,
    )


def four_circles(points=1000, test_size=0.3):
    """
    Create multiclass classification data. Creates 4 circles.

    :param points:
    Number of points in the dataset. Default = 1000
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    X = np.random.uniform(-2, 2, [points, 2])
    y = np.zeros(points, dtype=int) + 4
    centres = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    for j, c in enumerate(centres):
        for i, x in enumerate(X):
            if np.linalg.norm(x - c) <= 1:
                y[i] = j
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        2,
    )


def checkerboard(size=5, per_point=10, test_size=0.3):
    """
    Create binary distinguishable data. Data is in shape of a checkerboard with every other box being of opposite class.
    Setting all to single class will give 50% accuracy.
    Checkerboard is of dimension size*size
    Each box has 'per_point' points.

    :param size:
    Dimension of the checkerboard. Default = 5
    :param per_point:
    Number of points in each box. Default = 10
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    if size < 2:
        print("Small size")
        print("Setting size to 2")
        size = 2
    if per_point < 5:
        print("Too few points")
        print("Setting points per box to 5")
        per_point = 5
    x = y = size
    data = np.random.random((x * y * per_point, 2))
    labels = []
    point = 0
    while point < len(data):
        found = False
        for i in range(x):
            for j in range(y):
                if (
                        data[point][0] <= (i + 1) * 1 / x
                        and data[point][1] <= (j + 1) * 1 / y
                ):
                    if (i + j) % 2 == 0:
                        labels.append(0)
                    else:
                        labels.append(1)
                    point = point + 1
                    found = True
                if found:
                    break
            if found:
                break
        if found:
            continue
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        2,
    )


def polynomial(degree=3, points=100, limit=3, test_size=0.3):
    """
    Create a dataset for regression with y = x ^ degree

    :param degree:
    Degree of polynomial. Default = 3
    :param points:
    Number of points. Default = 100
    :param limit:
    x range set to -limit to limit. Default = 3
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    if points < 100:
        print("Too few points")
        print(f"Setting points to 100")
        points = 100
    X = np.linspace(-limit, limit, points)
    y = [x ** degree for x in X]
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train).reshape(-1, 1),
        np.asarray(X_test).reshape(-1, 1),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        1,
    )


def sine_wave(points=100, limit=2 * np.pi, test_size=0.3):
    """
    Create a dataset with y = sin(x)

    :param points:
    Number of points. Default = 100
    :param limit:
    x range set to -limit to limit. Default = 2pi
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    if points < 100:
        print("Too few points")
        print("Setting points to 100")
        points = 100
    X = np.linspace(-limit, limit, points)
    y = [np.sin(x) for x in X]
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train).reshape(-1, 1),
        np.asarray(X_test).reshape(-1, 1),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        1,
    )


def exponential(points=100, limit=2, test_size=0.3, power=1):
    """
    Create a dataset with y = exp(x * power)

    :param points:
    Number of points. Default = 100
    :param limit:
    x range set to -limit to limit. Default = 2
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :param power:
    y = exp(x * power). Default = 1
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    if points < 100:
        print("Too few points")
        print("Setting points to 100")
        points = 100
    X = np.linspace(-limit, limit, points)
    y = [np.exp(x * power) for x in X]
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train).reshape(-1, 1),
        np.asarray(X_test).reshape(-1, 1),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        1,
    )


def mod_x(points=100, limit=10, test_size=0.3):
    """
    Create a dataset with y = |x|

    :param points:
    Number of points. Default = 100
    :param limit:
    x range set to -limit to limit. Default = 10
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    if points < 100:
        print("Too few points")
        print("Setting points to 100")
        points = 100
    X = np.linspace(-limit, limit, points)
    y = [abs(x) for x in X]
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train).reshape(-1, 1),
        np.asarray(X_test).reshape(-1, 1),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        1,
    )


def blobs(points=100, features=2, centers=4, test_size=0.3):
    """
    Make blobs of points for classification. Each blob is a separate class.

    :param points:
    Number of points. Default = 100
    :param features:
    Dimension of each point. Default = 2
    :param centers:
    Number of classes. Default = 2
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    features: dimension of data
    """
    if points < centers * 50:
        print("Too few points")
        print(f"Setting points to {centers * 50}")
        points = centers * 50
    X, y = dt.make_blobs(
        n_samples=points, n_features=features, centers=centers, random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        features,
    )


def moons(points=100, test_size=0.3):
    """
    Make points for classification.

    :param points:
    Number of points. Default = 100
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    dimension of data
    """
    if points < 100:
        print("Too few points")
        print("Setting points to 100")
        points = 100
    X, y = dt.make_moons(n_samples=points, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        2,
    )


def boston_housing(test_size=0.3):
    """
    Returns the famous Boston Housing dataset for regression.

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 506
    Features = 5
    """
    X, y = dt.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=5)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X_train_std_pca.shape[-1],
    )


def regression_points(points=100, features=3, info_features=None, test_size=0.3):
    """
    Generate a random regression problem

    :param points:
    Number of points. Default = 100
    :param features:
    Number of features of data. (This is the data dimension)
    :param info_features:
    The number of features used to build the linear model used to generate the output.
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    if info_features is None:
        info_features = features
    X, y = dt.make_regression(
        n_samples=points,
        n_features=features,
        n_informative=info_features,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        features,
    )


def UCI_air_quality(test_size=0.3):
    """
    Attribute Information:
    0 Date (DD/MM/YYYY)
    1 Time (HH.MM.SS)
    2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
    3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
    4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
    5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
    6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
    7 True hourly averaged NOx concentration in ppb (reference analyzer)
    8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
    9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
    10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
    11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
    12 Temperature in Â°C
    13 Relative Humidity (%)
    14 AH Absolute Humidity

    Date and Time information is dropped
    All ground truth values are also dropped
    A random sample of 1000 data points is used

    X: Features
    y: C6H6 values
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 1000
    Features = 8
    """
    df = pd.read_csv("datasets/AirQualityUCI/AirQualityUCI.csv", sep=";")
    df = df.drop(
        [
            "Unnamed: 15",
            "Unnamed: 16",
            "Date",
            "Time",
            "CO(GT)",
            "NMHC(GT)",
            "NOx(GT)",
            "NO2(GT)",
        ],
        axis=1,
    )
    df = df.replace(-200, np.NaN).dropna()
    df = df.sample(1000, random_state=seed)
    X = df.drop(labels=["C6H6(GT)"], axis=1)
    y = df[["C6H6(GT)"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_auto_mpg(test_size=0.3):
    """
    Attribute Information:
    1. mpg: continuous
    2. cylinders: multi-valued discrete
    3. displacement: continuous
    4. horsepower: continuous
    5. weight: continuous
    6. acceleration: continuous
    7. model year: multi-valued discrete
    8. origin: multi-valued discrete
    9. car name: string (unique for each instance)

    X: Features
    y: target -> mpg values
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 397
    Features = 7
    """
    df = pd.read_csv("datasets/auto mpg/auto mpg data.csv", header=None)
    df = df.replace("?", np.NaN).dropna()
    y = df[[0]]
    X = df.drop(labels=[0, 8], axis=1)
    X = X.apply(pd.to_numeric, errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_automobile(test_size=0.3):
    """
       Attribute Information:
        Attribute:                Attribute Range:
        ------------------        -----------------------------------------------
     1. symboling:                -3, -2, -1, 0, 1, 2, 3.
     2. normalized-losses:        continuous from 65 to 256.
     3. make:                     alfa-romero, audi, bmw, chevrolet, dodge, honda,
                                  isuzu, jaguar, mazda, mercedes-benz, mercury,
                                  mitsubishi, nissan, peugot, plymouth, porsche,
                                  renault, saab, subaru, toyota, volkswagen, volvo
     4. fuel-type:                diesel, gas.
     5. aspiration:               std, turbo.
     6. num-of-doors:             four, two.
     7. body-style:               hardtop, wagon, sedan, hatchback, convertible.
     8. drive-wheels:             4wd, fwd, rwd.
     9. engine-location:          front, rear.
    10. wheel-base:               continuous from 86.6 120.9.
    11. length:                   continuous from 141.1 to 208.1.
    12. width:                    continuous from 60.3 to 72.3.
    13. height:                   continuous from 47.8 to 59.8.
    14. curb-weight:              continuous from 1488 to 4066.
    15. engine-type:              dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
    16. num-of-cylinders:         eight, five, four, six, three, twelve, two.
    17. engine-size:              continuous from 61 to 326.
    18. fuel-system:              1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
    19. bore:                     continuous from 2.54 to 3.94.
    20. stroke:                   continuous from 2.07 to 4.17.
    21. compression-ratio:        continuous from 7 to 23.
    22. horsepower:               continuous from 48 to 288.
    23. peak-rpm:                 continuous from 4150 to 6600.
    24. city-mpg:                 continuous from 13 to 49.
    25. highway-mpg:              continuous from 16 to 54.
    26. price:                    continuous from 5118 to 45400.

       All categorical features except num-of-cylinders are dropped.
       num-of-cylinders is converted to int

       X: Features
       y: Car price
       :param test_size:
       Size of testing dataset. Training dataset = 1 - test_size
       Default = 30% test and 70% train
       :return:
       X_train: Training features
       X_test: Testing features
       y_train: Training labels
       y_test: Testing labels
       X.shape[-1]: dimension of data
    """
    """
    Data points = 159
    Features = 7
    """
    df = pd.read_csv("datasets/autos/imports-85.data", header=None)
    df = df.replace("?", np.NaN).dropna()
    y = df[[25]]
    y = y.apply(pd.to_numeric)
    X = df.drop(labels=25, axis=1)
    X = X.apply(pd.to_numeric, errors="ignore")
    X = X.drop(labels=[17, 14, 3, 4, 5, 6, 7, 8, 2], axis=1)

    change_num = {
        "eight": 8,
        "four": 4,
        "five": 5,
        "six": 6,
        "three": 3,
        "twelve": 12,
        "two": 2,
    }

    def word_num(word):
        return change_num[word]

    X[15] = X[15].apply(word_num)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=7)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X_train_std_pca.shape[-1],
    )


def UCI_bike_share(test_size=0.3):
    """
    Only the day dataset is used here.

    Attribute Information:
    - instant: record index
    - dteday : date
    - season : season (1:springer, 2:summer, 3:fall, 4:winter)
    - yr : year (0: 2011, 1:2012)
    - mnth : month ( 1 to 12)
    - holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
    - weekday : day of the week
    - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
    + weathersit :
        - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
        - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
        - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    - temp : Normalized temperature in Celsius. The values are divided to 41 (max)
    - atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
    - hum: Normalized humidity. The values are divided to 100 (max)
    - windspeed: Normalized wind speed. The values are divided to 67 (max)
    - casual: count of casual users
    - registered: count of registered users
    - cnt: count of total rental bikes including both casual and registered

    Categorical features are dropped.

    X: Features
    y: Total number of bikes rented

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 731
    Features = 7
    """
    df = pd.read_csv("datasets/Bike-Sharing-Dataset/day.csv")
    y = df[["cnt"]]
    X = df.drop(labels=["cnt", "casual", "registered"], axis=1)
    X = X.astype(
        {"season": object, "mnth": object, "weekday": object, "weathersit": object}
    )
    X = X.drop(labels=["instant", "dteday"], axis=1)
    X = X.select_dtypes(["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_computer_hardware(test_size=0.3):
    """
    Attribute Information:
    1. vendor name: 30
      (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec,
       dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson,
       microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry,
       sratus, wang)
    2. Model Name: many unique symbols
    3. MYCT: machine cycle time in nanoseconds (integer)
    4. MMIN: minimum main memory in kilobytes (integer)
    5. MMAX: maximum main memory in kilobytes (integer)
    6. CACH: cache memory in kilobytes (integer)
    7. CHMIN: minimum channels in units (integer)
    8. CHMAX: maximum channels in units (integer)
    9. PRP: published relative performance (integer)
    10. ERP: estimated relative performance from the original article (integer)

    vendor name and Model Name and ERP are dropped

    X: Features
    y: published relative performance

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 209
    Features = 6
    """
    df = pd.read_csv("datasets/computer hardware/machine.data", header=None)
    y = df[[8]]
    X = df.drop(labels=[8, 9, 1, 0], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_energy_efficiency(test_size=0.3, data=1):
    """
    Attribute Information:
    X1 Relative Compactness
    X2 Surface Area
    X3 Wall Area
    X4 Roof Area
    X5 Overall Height
    X6 Orientation
    X7 Glazing Area
    X8 Glazing Area Distribution
    y1 Heating Load
    y2 Cooling Load

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :param data:
    There are two target variables, y1 and y2
    data=1 -> y1
    data=2 -> y2
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 768
    Features = 8
    """
    df = pd.read_excel("datasets/energy efficiency/ENB2012_data.xlsx")
    y = df[[f"Y{data}"]]
    X = df.drop(labels=["Y1", "Y2"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_forest_fires(test_size=0.3):
    """
    Attribute Information:
    1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
    2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
    3. month - month of the year: 'jan' to 'dec'
    4. day - day of the week: 'mon' to 'sun'
    5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
    6. DMC - DMC index from the FWI system: 1.1 to 291.3
    7. DC - DC index from the FWI system: 7.9 to 860.6
    8. ISI - ISI index from the FWI system: 0.0 to 56.10
    9. temp - temperature in Celsius degrees: 2.2 to 33.30
    10. RH - relative humidity in %: 15.0 to 100
    11. wind - wind speed in km/h: 0.40 to 9.40
    12. rain - outside rain in mm/m2 : 0.0 to 6.4
    13. area - the burned area of the forest (in ha): 0.00 to 1090.84

    Categorical features month and day are dropped

    X: Features
    y: area
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 517
    Features = 10
    """
    df = pd.read_csv("datasets/forest fires/forestfires.csv")
    y = df[["area"]]
    X = df.drop(labels=["area", "month", "day"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_student_performance(test_size=0.3, data=1):
    """
    Attribute Information:
    # Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:
    1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
    2 sex - student's sex (binary: 'F' - female or 'M' - male)
    3 age - student's age (numeric: from 15 to 22)
    4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
    5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
    6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
    7 Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
    8 Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
    9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
    10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
    11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
    12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
    13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
    14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
    15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
    16 schoolsup - extra educational support (binary: yes or no)
    17 famsup - family educational support (binary: yes or no)
    18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
    19 activities - extra-curricular activities (binary: yes or no)
    20 nursery - attended nursery school (binary: yes or no)
    21 higher - wants to take higher education (binary: yes or no)
    22 internet - Internet access at home (binary: yes or no)
    23 romantic - with a romantic relationship (binary: yes or no)
    24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
    25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
    26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
    27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
    28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
    29 health - current health status (numeric: from 1 - very bad to 5 - very good)
    30 absences - number of school absences (numeric: from 0 to 93)

    # these grades are related with the course subject, Math or Portuguese:
    31 G1 - first period grade (numeric: from 0 to 20)
    31 G2 - second period grade (numeric: from 0 to 20)
    32 G3 - final grade (numeric: from 0 to 20, output target)

    All categorical columns are dropped
    G1 and G2 grades are also dropped as they are highly correlated to G3

    X: Features
    y: G3 (final grade)

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :param data:
    There are two datasets:
    data=1 -> math dataset
    data=2 -> portuguese dataset
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data 1:
    Data Points = 395
    Features = 6
    Data 2:
    Data Points = 649
    Features = 6
    """
    if data == 1:
        df = pd.read_csv("datasets/student/student-mat.csv", sep=";").select_dtypes(
            ["number"]
        )
    else:
        df = pd.read_csv("datasets/student/student-por.csv", sep=";").select_dtypes(
            ["number"]
        )
    X = df.drop(labels=["G3", "G1", "G2"], axis=1)
    y = df[["G3"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=6)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X_train_std_pca.shape[-1],
    )


def UCI_wine_quality(test_size=0.3):
    """
    Attribute information:
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)

    X: Features
    y: quality

    A random sample of 1000 data points is used.

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 1000
    Features = 5
    """
    df = pd.read_csv("datasets/wine quality/winequality-red.csv", sep=";")
    df = df.sample(1000, random_state=seed)
    X = df.drop(labels=["quality"], axis=1)
    y = df[["quality"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=5)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X_train_std_pca.shape[-1],
    )


def UCI_abalone(test_size=0.3):
    """
    Attribute Information:
    Given is the attribute name, attribute type, the measurement unit and a
    brief description.  The number of rings is the value to predict: either
    as a continuous value or as a classification problem.

        Name		    Data Type	Meas.	Description
        ----		    ---------	-----	-----------
        Sex 		    nominal		M, F, and I (infant)
        Length		    continuous  mm      Longest shell measurement
        Diameter	    continuous	mm      perpendicular to length
        Height		    continuous	mm      with meat in shell
        Whole weight	continuous	grams	whole abalone
        Shucked weight	continuous	grams	weight of meat
        Viscera weight	continuous	grams	gut weight (after bleeding)
        Shell weight	continuous	grams	after being dried
        Rings		    integer             +1.5 gives the age in years

    Categorical feature Sex is dropped.
    A random sample of 1000 data points is used.

    X: Features
    y: Rings

    Rings is combined into 3 classes 1-8, 9-10, 11 and above.
    These 3 classes have similar number of data points.

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 1000
    Features = 7
    """
    data = pd.read_csv("datasets/Classification/abalone/abalone.data", header=None)
    data[8] = [0 if i < 9 else i for i in data[8]]
    data[8] = [1 if 9 <= i <= 10 else i for i in data[8]]
    data[8] = [2 if i > 10 else i for i in data[8]]
    data = data.sample(1000, random_state=seed)
    y = data[8]
    X = data.drop(labels=[0, 8], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_breast_cancer(test_size=0.3):
    """
     Attribute Information:

     #  Attribute                     Domain
     -- -----------------------------------------
     1. Sample code number            id number
     2. Clump Thickness               1 - 10
     3. Uniformity of Cell Size       1 - 10
     4. Uniformity of Cell Shape      1 - 10
     5. Marginal Adhesion             1 - 10
     6. Single Epithelial Cell Size   1 - 10
     7. Bare Nuclei                   1 - 10
     8. Bland Chromatin               1 - 10
     9. Normal Nucleoli               1 - 10
    10. Mitoses                       1 - 10
    11. Class:                        (2 for benign, 4 for malignant)

     id number is dropped.
     Rows with missing values are dropped.

     :param test_size:
     Size of testing dataset. Training dataset = 1 - test_size
     Default = 30% test and 70% train
     :return:
     X_train: Training features
     X_test: Testing features
     y_train: Training labels
     y_test: Testing labels
     X.shape[-1]: dimension of data
    """
    """
    Data Points = 693
    Features = 9
    """
    data = pd.read_csv(
        "datasets/Classification/breast_cancer/breast-cancer-wisconsin.data",
        header=None,
    )
    data = data.replace("?", np.NaN).dropna()
    y = data[10]
    y = y.replace(2, 0)
    y = y.replace(4, 1)
    y = y.to_numpy()
    X = data.drop(labels=[0, 10], axis=1)
    X = X.apply(pd.to_numeric)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        X.shape[-1],
    )


def UCI_car(test_size=0.3):
    """
    Attribute information:
    CAR                     car acceptability
    PRICE                   overall price
    buying                  buying price
    maint                   price of the maintenance
    TECH                    technical characteristics
    COMFORT                 comfort
    doors                   number of doors
    persons                 capacity in terms of persons to carry
    lug_boot                the size of luggage boot
    safety                  estimated safety of the car

    X: Features
    y: Safety 1-4

    Categorical columns are converted to numerical.
    A random sample of 1000 data points are used.

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 1000
    Features = 6
    """
    data = pd.read_csv("datasets/Classification/car/car.data", header=None)
    data.replace("low", "1", inplace=True)
    data.replace("med", "2", inplace=True)
    data.replace("high", "3", inplace=True)
    data.replace("vhigh", "4", inplace=True)
    data.replace("5more", "6", inplace=True)
    data.replace("more", "5", inplace=True)
    data.replace("small", "1", inplace=True)
    data.replace("big", "3", inplace=True)
    data.replace("unacc", "1", inplace=True)
    data.replace("acc", "2", inplace=True)
    data.replace("good", "3", inplace=True)
    data.replace("vgood", "4", inplace=True)
    data = data.apply(pd.to_numeric)
    data = data.sample(1000, random_state=seed)
    y = data[6]
    y = y - 1
    X = data.drop(labels=[6], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_heart(test_size=0.3):
    """
    Attribute Information:
    -- 1.   (age)
    -- 2.   (sex)
    -- 3.   (cp)
    -- 4.   (trestbps)
    -- 5.   (chol)
    -- 6.   (fbs)
    -- 7.   (restecg)
    -- 8.   (thalach)
    -- 9.   (exang)
    -- 10.  (oldpeak)
    -- 11.  (slope)
    -- 12.  (ca)
    -- 13.  (thal)
    -- 14.  (num)

    X: Features
    y: Heart Disease presence (1,2,3,4) and absence (0)

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 297
    Features = 5
    """
    data = pd.read_csv(
        "datasets/Classification/heart/processed.cleveland.data", header=None
    )
    data = data.replace("?", np.NaN).dropna()
    data = data.apply(pd.to_numeric)
    y = data[13]
    X = data.drop(labels=[13], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=5)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X_train_std_pca.shape[-1],
    )


def UCI_iris(test_size=0.3):
    """
    Attribute Information:
    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm
    5. class:
      -- Iris Setosa - 0
      -- Iris Versicolour - 1
      -- Iris Virginica - 2

    X: Features
    y: Flower type

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 150
    Features = 4
    """
    data = pd.read_csv("datasets/Classification/Iris/iris.data", header=None)
    data = data.replace("Iris-setosa", 0)
    data = data.replace("Iris-versicolor", 1)
    data = data.replace("Iris-virginica", 2)
    y = data[4]
    X = data.drop(labels=[4], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X.shape[-1],
    )


def UCI_wine(test_size=0.3):
    """
    Attribute Information:
    1) Alcohol
    2) Malic acid
    3) Ash
    4) Alcalinity of ash
    5) Magnesium
    6) Total phenols
    7) Flavanoids
    8) Nonflavanoid phenols
    9) Proanthocyanins
    10) Color intensity
    11) Hue
    12) OD280/OD315 of diluted wines
    13) Proline

    X: Features
    y: Class - Different cultivators

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 178
    Features = 5
    """
    data = pd.read_csv("datasets/Classification/Wine/wine.data", header=None)
    y = data[0]
    y = y - 1
    X = data.drop(labels=[0], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=5)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X_train_std_pca.shape[-1],
    )


def MNIST(points=1000, pca_dimension=6, test_size=0.3):
    """
    MNIST handwritten digits dataset.

    :param points:
    Size of sample dataset to use. Default = 1000
    :param pca_dimension:
    Number of principal components to use. Default = 6
    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X_train_std_pca.shape[-1]: dimension of data
    """
    df = pd.read_csv("datasets/Classification/MNIST/train.csv")
    df = df.sample(points, random_state=seed)
    y = df["label"]
    X = df.drop("label", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=pca_dimension)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train).reshape(-1, 1),
        np.asarray(y_test).reshape(-1, 1),
        X_train_std_pca.shape[-1],
    )


def UCI_bank(test_size=0.3):
    """
    Attribute Information:
    # bank client data:
    1 - age (numeric)
    2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
    3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
    4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
    5 - default: has credit in default? (categorical: "no","yes","unknown")
    6 - housing: has housing loan? (categorical: "no","yes","unknown")
    7 - loan: has personal loan? (categorical: "no","yes","unknown")
    # related with the last contact of the current campaign:
    8 - contact: contact communication type (categorical: "cellular","telephone")
    9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
    10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
    11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
    # other attributes:
    12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14 - previous: number of contacts performed before this campaign and for this client (numeric)
    15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
    # social and economic context attributes
    16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17 - cons.price.idx: consumer price index - monthly indicator (numeric)
    18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
    19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
    20 - nr.employed: number of employees - quarterly indicator (numeric)

    Output variable (desired target):
    21 - y - has the client subscribed a term deposit? (binary: "yes","no")

    Unknown values are removed.
    yes and no is mapped to 0 and 1 respectively.
    All other categorical features are removed.

    A random sample of 1000 data points are used.

    X: Features
    y: Has the client subscribed a term deposit?

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 1000
    Features = 5
    """
    df = pd.read_csv("datasets/Classification/bank/bank-additional.csv", sep=";")
    df = df.replace("unknown", np.NaN).dropna()
    df = df.replace("no", 1)
    df = df.replace("yes", 0)
    df = df.select_dtypes(["number"])
    df = df.sample(1000, random_state=seed)
    y = df["y"]
    y = y.to_numpy()
    X = df.drop(labels=["y", "duration"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=5)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train),
        np.asarray(y_test),
        X_train_std_pca.shape[-1],
    )


def UCI_Adult(test_size=0.3):
    """
    Attribute Information:
    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

    Sex and output label are converted into binary numerical.
    Other categorical features are removed.

    A random sample of 1000 data points is used.

    X: Features
    y: Whether a person makes over 50K a year.

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 1000
    Features = 7
    """
    df = pd.read_csv("datasets/Classification/Adult/adult.data", header=None)
    df = df.replace(" >50K", 1)
    df = df.replace(" <=50K", 0)
    df = df.replace(" Female", 0)
    df = df.replace(" Male", 1)
    df = df.select_dtypes(["number"])
    df = df.sample(1000, random_state=seed)
    y = df[14]
    y = y.to_numpy()
    X = df.drop(labels=[14], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        X.shape[-1],
    )


def UCI_wine_quality_classification(test_size=0.3):
    """
    Attribute information:
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)

    X: Features
    y: quality

    Only Red wine datest is used.
    Quality is converted into binary label. 0 for <= 5 and 1 for > 5
    A random sample of 1000 points is used.

    :param test_size:
    Size of testing dataset. Training dataset = 1 - test_size
    Default = 30% test and 70% train
    :return:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    X.shape[-1]: dimension of data
    """
    """
    Data Points = 1000
    Features = 6
    """
    df = pd.read_csv(
        "datasets/Classification/Wine_Qualtiy/winequality-red.csv", sep=";"
    )
    df["quality"] = [0 if i <= 5 else 1 for i in df["quality"]]
    df = df.sample(1000, random_state=seed)
    X = df.drop(labels=["quality"], axis=1)
    y = df[["quality"]]
    y = y.to_numpy()
    y = [i[0] for i in y]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pca = PCA(n_components=6)
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    return (
        np.asarray(X_train_std_pca),
        np.asarray(X_test_std_pca),
        np.asarray(y_train),
        np.asarray(y_test),
        X_train_std_pca.shape[-1],
    )
