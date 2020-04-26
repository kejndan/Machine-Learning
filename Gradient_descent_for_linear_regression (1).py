import numpy as np
import matplotlib.pyplot as plt
from random import randint


def design_matrix(x, n):
    phi_n = np.empty((len(x), n + 1))
    phi_n[:, 0] = 1
    phi_n[:, 1] = x
    for i in range(2, n + 1):
        phi_n[:, i] = phi_n[:, i - 1] * phi_n[:, 1]
    return phi_n


def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / len(y_true)


def draw_gist(confidences, lambdas):
    number_bins = len(lambdas)
    labels = []
    width = 0.3
    confidences_train = confidences[0]
    confidences_valid = confidences[1]
    for i in range(number_bins):
        reg = "".join(f"{lambdas[i]}")
        labels.append(reg)
    bin_positions = np.array(list(range(len(confidences_train))))
    fig, ax = plt.subplots(figsize=(15, 5))
    rects1 = ax.bar(bin_positions - width / 2, confidences_train, width)
    rects2 = ax.bar(bin_positions + width / 2, confidences_valid, width)
    plt.ylabel('Ошибка')
    plt.title('Ошибки на различных архитектурах')
    plt.xticks(bin_positions, labels)
    plt.legend(loc=3)
    for rect in rects1 :
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, f'{height:.2f}', ha='center', va='bottom')
    for rect in rects2 :
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, f'{height:.2f}', ha='center', va='bottom')
    plt.show()


def split_train_valid_test_with_max_dispersion(data, orig_y, train_size=0.8, valid_size=0.1, test_size=0.1):
    mask = np.concatenate((np.arange(len(data))[:, np.newaxis], abs(data[:, -1:]-orig_y[:, np.newaxis])), axis=1)
    mask = np.array(sorted(mask, key=lambda x: x[-1]))[::-1]
    data = data[np.round(mask[:, 0], 0).astype(int).T]
    len_train = int(train_size * len(data))
    len_valid = int(valid_size * len(data))
    return data[:len_train], data[len_train :len_train + len_valid], data[len_train + len_valid:]


def loss(x, t, w, lamb, n):
    e_d = np.power(t - np.dot(w, design_matrix(x, n).T), 2).sum() / 2
    e_r = lamb * np.power(w, 2).sum() / 2
    return e_d + e_r


def gradient(x, t, w, lamb, n):
    temp = -(t - np.dot(w, design_matrix(x, n).T)).dot(design_matrix(x, n)) + lamb * w
    return temp


def gradient_descent(x, t, n, step, lamb) :
    loss_values = []
    w_next = np.random.rand(n + 1).reshape((1, n + 1)) / 100
    cant_stop = True
    count = 0
    while cant_stop:
        w_old = w_next
        w_next = w_old - step * gradient(x, t, w_old, lamb, n)
        loss_values.append(loss(x, t, w_next, lamb, n))
        if np.linalg.norm(w_old - w_next) < eps * np.linalg.norm(w_next) + eps0:
            cant_stop = False
        count += 1
        print(loss_values[-1], count)
    return loss_values, w_next


if __name__ == '__main__':
    np.random.seed(665)
    eps = 0.001
    eps0 = 0.0001
    step = 0.00005
    lambdas_reg = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 5, 10, 20, 30, 40, 50, 60, 70]


    points = 100
    poly_deg = 15
    x = np.linspace(0, 1, points)
    gt = 30 * x * x
    err = 2 * np.random.randn(points)
    err[3] += 200
    err[77] += 100
    err[50] -= 100
    t = gt + err
    orig_data = np.concatenate((x[:, np.newaxis], t[:, np.newaxis]), axis=1)
    train_data, valid_data, test_data = split_train_valid_test_with_max_dispersion(orig_data.copy(), gt)
    x_train, t_train = train_data[:, 0], train_data[:, 1]
    x_valid, t_valid = valid_data[:, 0], valid_data[:, 1]
    x_test, t_test = test_data[:, 0], test_data[:, 1]
    confidences_train = []
    confidences_valid = []
    weights = []
    for i, lmbd_reg in enumerate(lambdas_reg):
        loss_values, w_final = gradient_descent(x_train, t_train, poly_deg, step, lmbd_reg)
        confidences_train.append(mean_squared_error(t_train, w_final.dot(design_matrix(x_train, poly_deg).T)))
        confidences_valid.append(mean_squared_error(t_valid, w_final.dot(design_matrix(x_valid, poly_deg).T)))
        weights.append(w_final)
        if lmbd_reg == 0 or lmbd_reg == 70:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(x_train, t_train, 'ro', markersize=3)
            ax1.plot(x_train, w_final.dot(design_matrix(x_train, poly_deg).T).flatten(), 'bo', markersize=3)
            ax2.plot(list(range(1, len(loss_values)+1)), loss_values)
            ax1.set_ylim(-100, 100)
            ax2.set_xlabel("Итерация")
            ax2.set_ylabel("Ошибка")
            plt.show()
    draw_gist([confidences_train, confidences_valid], lambdas_reg)
    sorted_confidences_valid = sorted(list(enumerate(confidences_valid)), key=lambda x: x[1])
    # print(sorted_confidences_valid)
    w_best = weights[sorted_confidences_valid[8][0]]
    mse_test = mean_squared_error(t_test, w_best.dot(design_matrix(x_test, poly_deg).T).flatten())
    fig, ax1 = plt.subplots()
    ax1.plot(x, t, 'ro', markersize=3)
    ax1.plot(x, w_best.dot(design_matrix(x, poly_deg).T).flatten(), 'bo', markersize=3)
    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax1.set_title(f"lambda:{lambdas_reg[sorted_confidences_valid[0][0]]}\n Ошибка на тестовой: {mse_test}")
    plt.show()




