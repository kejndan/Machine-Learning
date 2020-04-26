import numpy as np
from random import randint
from matplotlib import pyplot as plt

def random_classifier(x):
    return randint(0, 1)


def height_classifier(x, height):
    if x > height:
        return 1
    else:
        return 0


def split_train_valid_test(data, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Функция разделения выборки
    """
    np.random.shuffle(data)
    len_train = int(train_size*len(data))
    len_valid = int(valid_size*len(data))
    return data[:len_train], data[len_train:len_train+len_valid], data[len_train+len_valid:]


def get_type_result(y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        return 'TP'
    elif y_true == 0 and y_pred == 0:
        return 'TN'
    elif y_true == 0 and y_pred == 1:
        return 'FP'
    elif y_true == 1 and y_pred == 0:
        return 'FN'



def accuracy(true_positives, true_negatives, n):
    return (true_positives+true_negatives)/n



def precision(true_positives, false_positives):
    try:
        return true_positives/(true_positives+false_positives)
    except ZeroDivisionError:
        return 1


def recall(true_positives, false_negatives):
    try:
        return true_positives/(true_positives+false_negatives)
    except ZeroDivisionError:
        print('Данные не содержат класс 1')


if __name__ == '__main__':
    np.random.seed(80)
    height_basketball_player = np.random.randn(500) * 10 + 190  # class 1
    height_football_player = np.random.randn(500) * 20 + 160  # class 0
    # print(height_football_player.max())
    # print(height_basketball_player.max())
    data_football_player = np.concatenate((height_football_player[:, None], np.zeros(500)[:, None]), axis=1)
    data_basketball_player = np.concatenate((height_basketball_player[:, None], np.ones(500)[:, None]), axis=1)
    full_data = np.concatenate((data_basketball_player, data_football_player))
    np.random.shuffle(full_data)
    # print(full_data[np.argmax(full_data[:,0])])
    res_precision_height = []
    res_recall_height = []
    for height in range(80, 231, 10):
        results = {'random' : {'TP' : 0, 'TN' : 0, 'FP' : 0, 'FN' : 0},
                   'height' : {'TP' : 0, 'TN' : 0, 'FP' : 0, 'FN' : 0}}
        for i in range(len(full_data)):
            y_random = random_classifier(full_data[i, 0])
            y_height = height_classifier(full_data[i, 0], height)
            results['random'][get_type_result(full_data[i, 1], y_random)] += 1
            results['height'][get_type_result(full_data[i, 1], y_height)] += 1

        accuracy_random = accuracy(results['random']['TP'], results['random']['TN'], 1000)
        precision_random = precision(results['random']['TP'], results['random']['FP'])
        recall_random = recall(results['random']['TP'], results['random']['FN'])

        accuracy_height = accuracy(results['height']['TP'], results['height']['TN'], 1000)
        precision_height = precision(results['height']['TP'], results['height']['FP'])
        recall_height = recall(results['height']['TP'], results['height']['FN'])
        res_precision_height.append(precision_height)
        res_recall_height.append(recall_height)

        print(f"Random: accuracy:{accuracy_random}, precision:{precision_random}, recall:{recall_random}")
        print(f"Height: {height}  accuracy:{accuracy_height}, precision:{precision_height}, recall:{recall_height}")

    res_recall_height = [0]+res_recall_height[::-1]+[1]
    res_precision_height = [1]+res_precision_height[::-1]+[0]

    s = 0
    for i in range(1, len(res_recall_height)):
        s += (res_recall_height[i]-res_recall_height[i-1])*(res_precision_height[i]+res_precision_height[i-1])/2
    plt.plot(res_recall_height, res_precision_height, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Square:{s}')
    plt.show()












