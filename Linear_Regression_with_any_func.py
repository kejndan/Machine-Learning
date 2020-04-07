import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


def split_train_valid_test(data, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Функция разделения выборки
    """
    np.random.shuffle(data)
    len_train = int(train_size*len(data))
    len_valid = int(valid_size*len(data))
    return data[:len_train], data[len_train:len_train+len_valid], data[len_train+len_valid:]


def func_to_numpy(func, x):
    return eval(func)

# Генерация выборки
n = 1000
x = np.linspace(0.00001,2*np.pi, n)
y = 100*np.sin(x) + 0.5*np.exp(x) + 300
err = np.random.rand(n)
y = y + err
# Разбиение выборки
orig_data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)
train_data, valid_data, test_data = split_train_valid_test(orig_data.copy())


basis_function = np.array(['np.sin(x)', 'np.cos(x)', 'np.exp(x)', 'np.log(x)',
                  'np.sqrt(x)', 'np.power(x,1)', 'np.power(x,2)', 'np.power(x,3)'])

x_train, y_train = train_data[:, 0], train_data[:, 1]
x_valid, y_valid = valid_data[:, 0], valid_data[:, 1]
x_test, y_test = test_data[:, 0], test_data[:, 1]



list_models = []
for k in range(1, 4):
    for comb in combinations(np.arange(len(basis_function)), k):
        # Определяем веса для модели
        phi1 = np.array([1 for _ in range(len(x_train))])[:, np.newaxis]
        phi = phi1.copy()
        for i1 in comb:
            phi = np.concatenate((phi, func_to_numpy(basis_function[i1], x_train[:, np.newaxis])),
                                 axis=1)
        y1 = np.dot(phi.T, phi)
        y2 = np.linalg.inv(y1)
        y3 = y2.dot(phi.T)
        phi2 = np.linalg.inv(np.dot(phi.T, phi)).dot(phi.T)
        w = phi2.dot(y_train)
        new_y = (w * phi).sum(axis=1)
        # Считаем MSE для обучающей модели
        mse_train = ((y_train - new_y) ** 2).sum() / len(x_train)
        # Считаем ошибку на валидационной выборке
        phi1_valid = np.array([1 for _ in range(len(x_valid))])[:, np.newaxis]
        phi = phi1_valid.copy()
        for i1 in comb:
            phi = np.concatenate((phi, func_to_numpy(basis_function[i1], x_valid[:, np.newaxis])),
                                 axis=1)
        new_y_valid = (w * phi).sum(axis=1)
        mse_valid = ((y_valid - new_y_valid) ** 2).sum() / len(x_valid)
        # Сохраняем информацию о модели
        models = dict()
        models['Model'] = list(comb)
        models['MSE_TRAIN'] = mse_train
        models['MSE_VALID'] = mse_valid
        models['W'] = w
        list_models.append(models)
list_models = sorted(list_models, key=lambda x: x['MSE_VALID']) # Сортируем модели по валидационной ошибке
for i in range(3):
    comb = list_models[i]['Model']
    w = list_models[i]['W']
    # Считаем ошибку на тестовой выборке
    phi1_test = np.array([1 for _ in range(len(x_test))])[:, np.newaxis]
    phi = phi1_test.copy()
    for i1 in comb:
        phi = np.concatenate((phi, func_to_numpy(basis_function[i1], x_test[:, np.newaxis])),
                     axis=1)
    new_y_test = (w * phi).sum(axis=1)
    mse_test = ((y_test - new_y_test) ** 2).sum() / len(x_test)
    list_models[i]['MSE_TEST'] = mse_test
    # Считаем `y` через phi для всей выборки и рисуем график
    phi1_all = np.array([1 for _ in range(len(x))])[:, np.newaxis]
    phi = phi1_all.copy()
    for i1 in comb:
        phi = np.concatenate((phi, func_to_numpy(basis_function[i1], x[:, np.newaxis])),
                     axis=1)
    new_y_all = (w * phi).sum(axis=1)
    fig, ax = plt.subplots()
    label1 = 'Оригинал'
    label2 = "".join([f"{w:.2f}*{name}+" for w, name in zip(list_models[i]['W'][1:],basis_function[list_models[i]['Model']])]) +f"{list_models[i]['W'][0]:.2f}"
    ax.scatter(x=x,y = y, c='r',marker='o', label=label1)
    ax.scatter(x=x,y = new_y_all, c='g',marker='o',label=label2)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    # ax.set_title()
    ax.legend()
    plt.show()
# Заменяем номера базисных функций на их названия
for i, model in enumerate(list_models):
    for j, num in enumerate(model['Model']):
        list_models[i]['Model'][j] = basis_function[num]
# Печатаем 3 лучших модели
for i,e in enumerate(list_models[:3]):
    print('Модель',i+1)
    for k,v in e.items():
        print(k,v)




# Строим гистограмму
number_bins = 3
labels = []
width = 0.3
confidences_train = [model['MSE_TRAIN'] for model in list_models[:number_bins]]
confidences_valid = [model['MSE_VALID'] for model in list_models[:number_bins]]
for i in range(number_bins):
    weights= list_models[i]['W']
    func_names=list_models[i]['Model']
    reg = "".join([f"{w:.2f}*{name}+" for w, name in zip(weights[1:],func_names)])+f"{weights[0]:.2f}"
    labels.append(reg)
bin_positions=np.array(list(range(len(confidences_train))))
fig, ax = plt.subplots(figsize=(15,5))
rects1 = ax.bar(bin_positions-width/2, confidences_train, width, label="Точность на обучающей выборке")
rects2 = ax.bar(bin_positions+width/2, confidences_valid, width, label="Точность на обучающей валидационной")
plt.ylabel('Точность')
plt.title('Точность различных архитектур')
plt.xticks(bin_positions, labels)
plt.legend(loc=3)
for rect in rects1:
   height = rect.get_height()
   plt.text(rect.get_x() + rect.get_width()/2., 1.01*height, f'{height:.2f}', ha='center', va='bottom')
for rect in rects2:
   height = rect.get_height()
   plt.text(rect.get_x() + rect.get_width()/2., 1.01*height, f'{height:.2f}', ha='center', va='bottom')
plt.show()