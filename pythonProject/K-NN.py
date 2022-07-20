import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

training_data_set = []
testing_data_set = []
number_of_k = []

data = genfromtxt('iris.csv', delimiter=',')


def square(x):
    return x * x


def prediction(x):
    true_prediction_training = 0
    true_prediction_testing = 0

    random = shuffle(data)
    training_data, testing_data = train_test_split(random, test_size=0.2, train_size=0.8)

    for i in range(0, len(training_data), 1):
        array_dist_training = []
        array_dist_testing = []

        for j in range(0, len(training_data), 1):
            if j != i:
                dist0 = training_data[i][0] - training_data[j][0]
                dist1 = training_data[i][1] - training_data[j][1]
                dist2 = training_data[i][2] - training_data[j][2]
                dist3 = training_data[i][3] - training_data[j][3]

                dist_training = math.sqrt(square(dist0) + square(dist1) + square(dist2) + square(dist3))
                array_dist_training.append((dist_training, training_data[j][4]))

            if i < len(testing_data):
                dist00 = testing_data[i][0] - training_data[j][0]
                dist01 = testing_data[i][1] - training_data[j][1]
                dist02 = testing_data[i][2] - training_data[j][2]
                dist03 = testing_data[i][3] - training_data[j][3]

                dist_testing = math.sqrt(square(dist00) + square(dist01) + square(dist02) + square(dist03))
                array_dist_testing.append((dist_testing, training_data[j][4]))

        first_group_training, second_group_training, third_group_training = 0, 0, 0
        first_group_testing, second_group_testing, third_group_testing = 0, 0, 0

        array_dist_training.sort()
        array_dist_testing.sort()

        group_training, group_testing = 0, 0

        for k in range(0, x, 1):
            if array_dist_training[k][1] == 1:
                first_group_training += 1
            elif array_dist_training[k][1] == 2:
                second_group_training += 1
            elif array_dist_training[k][1] == 3:
                third_group_training += 1

            if i < len(testing_data):
                if array_dist_testing[k][1] == 1:
                    first_group_testing += 1
                elif array_dist_testing[k][1] == 2:
                    second_group_testing += 1
                elif array_dist_testing[k][1] == 3:
                    third_group_testing += 1

        if first_group_training > second_group_training and first_group_training > third_group_training:
            group_training = 1.0
        elif second_group_training > first_group_training and second_group_training > third_group_training:
            group_training = 2.0
        elif third_group_training > second_group_training and third_group_training > first_group_training:
            group_training = 3.0

        if i < 30:
            if first_group_testing >= second_group_testing and first_group_testing >= third_group_testing:
                group_testing = 1.0
            elif second_group_testing >= first_group_testing and second_group_testing >= third_group_testing:
                group_testing = 2.0
            elif third_group_testing >= second_group_testing and third_group_testing >= first_group_testing:
                group_testing = 3.0

        if training_data[i][4] == group_training:
            true_prediction_training += 1

        if i < 30:
            if testing_data[i][4] == group_testing:
                true_prediction_testing += 1

    return true_prediction_training / 120, true_prediction_testing / 30

standard_deviation_training = []
standard_deviation_testing = []

for i in range(0, 26, 1):
    number_of_k.append(i * 2 + 1)
    average_training = 0
    average_testing = 0

    train_data = []
    test_data = []

    for y in range(1, 20, 1):
        a, b = prediction(i * 2 + 1)
        train_data.append(a)
        test_data.append(b)

        average_training = average_training + a
        average_testing = average_testing + b

    standard_deviation_training.append(np.std(train_data))
    standard_deviation_testing.append(np.std(test_data))

    training_data_set.append(average_training / 20)
    testing_data_set.append(average_testing / 20)


plt.plot(number_of_k, training_data_set, color='blue', linewidth=1, marker='o', markerfacecolor='black', markersize=3)
plt.errorbar(number_of_k, training_data_set, yerr=standard_deviation_training)

plt.ylim(0.8, 1)
plt.xlim(0, 55)

plt.xlabel('(Value of k)')
# naming the y axis
plt.ylabel('(Accuracy over training data)')
plt.show()

plt.plot(number_of_k, testing_data_set, color='blue', linewidth=1, marker='o', markerfacecolor='black', markersize=3)
plt.errorbar(number_of_k, testing_data_set, yerr=standard_deviation_testing)
plt.ylim(0.8, 1)
plt.xlim(0, 55)

# naming the x axis
plt.xlabel('(Value of k)')
# naming the y axis
plt.ylabel('(Accuracy over testing data)')
plt.show()
