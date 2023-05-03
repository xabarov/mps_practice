import rand_destribution as rd
import numpy as np
import matplotlib.pyplot as plt
import network_calc
from network_im_prty import NetworkPrty
from network_viewer import show_DG, create_DG

def print_results(n, loads, k_num, v_im, v_ch):
    """
    Вывод результатов расчета
    nn: 
    loads: 
    k_num: 
    v_im: 
    v_ch: 
    """
    print("\n")
    print("-" * 60)
    print("{0:^60s}\n{1:^60s}".format("Сравнение данных ИМ и результатов расчета времени пребывания",
                                      "в СеМО с многоканальными узлами и приоритетами"))
    print("-" * 60)
    print("Количество каналов в узлах:")
    for nn in n:
        print("{0:^1d}".format(nn), end=" ")
    print("\nКоэффициенты загрузки узлов :")
    for load in loads:
        print("{0:^1.3f}".format(load), end=" ")
    print("\n")
    print("-" * 60)
    print("{0:^60s}".format("Относительный приоритет"))

    print("-" * 60)
    print("{0:^11s}|{1:^47s}|".format('', 'Номер начального момента'))
    print("{0:^10s}| ".format('№ кл'), end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(3):
        s = str(j + 1)
        print("{:^15s}|".format(s), end="")
    print("")
    print("-" * 60)

    for i in range(k_num):
        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("ИМ"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_im[i][j]), end="")
        print("")
        print("{:^5s}".format(str(i + 1)) + "|" + "-" * 54)

        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("Р"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_ch[i][j]), end="")
        print("")
        print("-" * 60)

    print("\n")


k_num = 3  # кол-во классов заявок
n_num = 5  # кол-во узлов сети
n = [3, 2, 3, 4, 3]  # кол-во каналов обслуживания в узлах

coev_fix = 0.7  # коэфф вариации времени обслуживания в узлах (при построении зависимости от ro)
ro_fix = 0.7  # фиксированный коэфф загрузки в каждом узле (при построении зависимости от coev)

is_np = True  # False - абсолютный приоритет, True - относительный

L = [0.9 / k_num] * k_num  # интенсивности поступления заявок в сеть

jobs_num = 30000  # кол-во требуемых к обслуживанию заявок в ИМ сети

#  списки для накопления времен пребывания в сети
vs_ch = []  # для численного расчета
vs_im = []  # для ИМ

for j in range(k_num):
    vs_ch.append([])
    vs_im.append([])

#  коэфф загрузки узлов. Для построения зависимости от коэфф загрузки узлов в сети
roes = np.linspace(0.1, 0.9, 15)

# Матрицы интенсивностей переходов
R = []

for i in range(k_num):
    R.append(np.matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 0.4, 0.6, 0, 0, 0],
        [0, 0, 0, 0.6, 0.4, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]))

DG = create_DG(R[0])
show_DG(DG)
# зависимость от коэфф загрузки сети
for ro in roes:

    b = []  # список начальных моментов обслуживания в каждом из узлов по классам
    # [k][node][j], k - номер класса, node - номер узла, j - номер начального момента

    serv_params = []  # параметры распределения обслуживания [node][k]{'type': type, 'params': params}
    gamma_params = []  # параметры Гамма-распределения [node][mu, alpha]
    nodes_prty = []  # приоритеты по узлам сети [node][номера приоритетов по классам]

    for m in range(n_num):
        nodes_prty.append([])
        for j in range(k_num):
            nodes_prty[m].append(j)
            # if m == 4:
            #     nodes_prty[m].append(k_num - j - 1)

        b1 = ro * n[m] / sum(L)
        gamma_params.append(rd.Gamma.get_mu_alpha_by_mean_and_coev(b1, coev_fix))

        serv_params.append([])
        for i in range(k_num):
            serv_params[m].append({'type': 'Gamma', 'params': gamma_params[m]})

    for k in range(k_num):
        b.append([])
        for m in range(n_num):
            b[k].append(rd.Gamma.calc_theory_moments(*gamma_params[m], 4))

    # Задаем тип приоритета по узлам. NP - относительный, PR - абсолютный
    # В нашем случае тип приоритета в каждом узле одинаковый
    if is_np:
        prty = ['NP'] * n_num
    else:
        prty = ['PR'] * n_num

    # Задаем ИМ сети
    semo_im = NetworkPrty(k_num, L, R, n, prty, serv_params, nodes_prty)
    # Запускаем на расчет
    semo_im.run(jobs_num)
    # Получаем результаты - начальные моменты времени пребывания в сети
    v_im = semo_im.v_semo

    # Численный расчет сети
    semo_calc = network_calc.network_prty_calc(R, b, n, L, prty, nodes_prty)
    v_ch = semo_calc['v']
    loads = semo_calc['loads']

    for j in range(k_num):
        vs_im[j].append(v_im[j][0])
        vs_ch[j].append(v_ch[j][0])

    # Вывод результатов расчета
    print_results(n, loads, k_num, v_im, v_ch)

# строим график от ro
fig, ax = plt.subplots()

for i in range(k_num):
    ax.plot(roes, vs_im[i], label="ИМ класс {0:d}".format(i + 1))
    ax.plot(roes, vs_ch[i], label="Числ класс {0:d}".format(i + 1))
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$\upsilon_{1}$")

if is_np:
    ax.set_title("Относительный приоритет")
else:
    ax.set_title("Абсолютный приоритет")

plt.legend()
plt.show()

coevs = np.linspace(0.3, 3.0, 15)

for coev in coevs:
    pass

# # строим график от coevs
# fig, ax = plt.subplots()
#
# for i in range(k_num):
#     ax.plot(coevs, vs_im[i], label="ИМ класс {0:d}".format(i+1))
#     ax.plot(coevs, vs_ch[i], label="Числ класс {0:d}".format(i+1))
# ax.set_xlabel(r"$\nu$")
# ax.set_ylabel(r"$\upsilon_{1}$")
#
# if is_np:
#     ax.set_title("Относительный приоритет")
# else:
#     ax.set_title("Абсолютный приоритет")
#
# plt.legend()
# plt.show()
