import math

import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = genfromtxt('house_votes_84.csv', delimiter=',', skip_header=1)


class TreeNode:
    def __init__(self , val):
        self.val = val
        self.left = None
        self.mid = None
        self.right = None


def In(a, b):
    if a == 0 or b == 0:
        return 0
    return -a * math.log2(a) - b * math.log2(b)


def checkPrediction(array, tree):
    if tree.val == 1000:
        if array[16] == 0:
            return 1
        else:
            return 0

    if tree.val == 2000:
        if array[16] == 1:
            return 1
        else:
            return 0

    value = tree.val
    if array[value] == 0 and tree.left is not None:
        return checkPrediction(array, tree.left)
    if array[value] == 1 and tree.mid is not None:
        return checkPrediction(array, tree.mid)
    if array[value] == 2 and tree.right is not None:
        return checkPrediction(array, tree.right)
    return 0

def buildDecisionTree(random):
    ans = 0
    val = 10

    for i in range(0, 16, 1):

        first_choice = 0
        second_choice = 0
        third_choice = 0
        nums00 = 0
        nums01 = 0
        nums02 = 0

        for j in range(0, len(random), 1):
            if random[j][i] == 0:
                first_choice += 1
                if random[j][16] == 0:
                    nums00 += 1

            elif random[j][i] == 1:
                second_choice += 1
                if random[j][16] == 0:
                    nums01 += 1

            else:
                third_choice += 1
                if random[j][16] == 0:
                    nums02 += 1

        res = 0
        if first_choice > 0:
            res += first_choice / len(random) * In(nums00 / first_choice,(first_choice - nums00) / first_choice)

        if second_choice > 0:
            res += second_choice / len(random) * In(nums01 / second_choice, (second_choice - nums01) / second_choice)

        if third_choice > 0:
            res += third_choice / len(random) * In(nums02 / third_choice,(third_choice - nums02) / third_choice)

        if res < val:
            val = res
            ans = i

    if (val < 1e-7):
        choice_0 = 0
        choice_1 = 0

        for i in range(0, len(random), 1):
            if random[i][16] == 0:
                choice_0 += 1
            else:
                choice_1 += 1

        if choice_0 > choice_1:
            node = TreeNode(1000)
            node.left = None
            node.right = None
        else:
            node = TreeNode(2000)
            node.left = None
            node.right = None
        return node

    node = TreeNode(ans)

    arr_left = []
    arr_mid = []
    arr_right = []

    for i in range(0, len(random), 1):
        if random[i][ans] == 0:
            arr_left.append(random[i])
        elif random[i][ans] == 1:
            arr_mid.append(random[i])
        elif random[i][ans] == 2:
            arr_right.append(random[i])

    if len(arr_left) > 0:
        node.left = buildDecisionTree(arr_left)
    if len(arr_mid) > 0:
        node.mid = buildDecisionTree(arr_mid)
    if len(arr_right) > 0:
        node.right = buildDecisionTree(arr_right)

    return node


def prediction():
    true_prediction_training = 0
    true_prediction_testing = 0

    random = shuffle(data)
    training_data, testing_data = train_test_split(random, test_size=0.2, train_size=0.8)


    tree = buildDecisionTree(training_data)

    for i in range(0, 348, 1):
        if checkPrediction(training_data[i], tree) == 1:
            true_prediction_training += 1
        if i < 87:
            if checkPrediction(testing_data[i], tree) == 1:
                true_prediction_testing += 1

    return true_prediction_training / 348, true_prediction_testing / 87


mp_testing = {}
mp_training = {}

for i in range(1, 101, 1):
    a, b = prediction()
    a_rounded = round(a, 2)
    b_rounded = round(b, 2)

    if a_rounded in mp_training.keys():
        x = mp_training[a_rounded]
        mp_training[a_rounded] = x + 1
    else:
        mp_training[a_rounded] = 1

    if b_rounded in mp_testing.keys():
        mp_testing[b_rounded] += + 1
    else:
        mp_testing[b_rounded] = 1

accuracy_training = list(mp_training.keys())
frequency_training = list(mp_training.values())

fig = plt.figure(figsize=(10, 5))
plt.bar(accuracy_training, frequency_training, width=0.007, align="center")
plt.xlabel("Accuracy")
plt.ylabel("Accuracy frequency of training data")
plt.show()

accuracy_testing = list(mp_testing.keys())
frequency_testing = list(mp_testing.values())

fig = plt.figure(figsize=(10, 5))
plt.bar(accuracy_testing, frequency_testing, width=0.007, align="center")
plt.xlabel("Accuracy")
plt.ylabel("Accuracy frequency of testing data")

plt.show()
