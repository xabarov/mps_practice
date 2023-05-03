import rand_destribution as rd
import numpy as np
import matplotlib.pyplot as plt
import prty_calc
import math
from smo_im_prty import SmoImPrty


def print_results(v_im, v_ch, k, ro, coev, is_np=True):
    """
    Вывод результатов расчетов и ИМ
    """
    print("\nСравнение данных ИМ и результатов расчета (Р) \n"
          "времени пребывания в многоканальной СМО с приоритетами")
    print("Число каналов: " + str(1) + "\nЧисло классов: " + str(k) + "\nКоэффициент загрузки: {0:<1.2f}".format(ro) +
          "\nКоэффициент вариации времени обслуживания: {0:3.3f}\n".format(coev))

    if is_np:
        print("Относитльный приоритет")
    else:
        print("Абсолютный приоритет")

    print("\nЗначения начальных моментов времени ожидания заявок в системе для ro = {0:1.3f}:\n".format(ro))

    print("-" * 45)
    print("{0:^11s}|{1:^31s}|".format('', 'Номер начального момента'))
    print("{0:^11s}| ".format('№ кл'), end="")
    print("-" * 29 + " |")

    print(" " * 11 + "|", end="")
    for j in range(2):
        s = str(j + 1)
        print("{:^15s}|".format(s), end="")
    print("")
    print("-" * 45)

    for i in range(k):
        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("ИМ"), end="")
        for j in range(2):
            print("{:^15.3g}|".format(v_im[i][j]), end="")
        print("")
        print("{:^5s}".format(str(i + 1)) + "|" + "-" * 39)

        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("Р"), end="")
        for j in range(2):
            print("{:^15.3g}|".format(v_ch[i][j]), end="")
        print("")
        print("-" * 45)


def get_b_by_mean_and_coev(mean, coev):
    """
    Аппроксимация Гамма-распределением.
    По заданному среднему и коэфф вариации возвращает 4 начальных момента
    """
    mu, alpha = rd.Gamma.get_mu_alpha_by_mean_and_coev(mean, coev)
    return rd.Gamma.calc_theory_moments(mu, alpha, 4)


# фиксированные значения из таблицы
ro_fix = 0.7
coev_fix = 1.2
k_fix = 2

is_np = True  # True - относительный приоритет, False - абсолютный

# задаем массив roes коээфициентов загрузки СМО от 0.1 до 0.9, 15 значений
roes = np.linspace(0.1, 0.9, 12)

# число заявок, которые будут обслужены в ИМ
num_of_jobs = 300000

# массивы для накопления средних времен пребывания в системе для численного метода и ИМ

vs_ch = []
vs_im = []

for j in range(k_fix):
    vs_ch.append([])
    vs_im.append([])

# зависимость от ro

for ro in roes:

    k = k_fix  # количество классов заявок
    ls = [1.0 / k] * k  # интенсивности поступления заявок, суммарная интенсивность = 1

    # Начальные моменты времени обслуживания в СМО для каждого из классов
    bs = []
    for j in range(k):
        # Аппроксимируем Гамма-распределением по заданным МО и коэфф вариации.
        # МО == коэфф загрузки для одноканальной СМО при суммарной интенсивности поступления заявок = 1
        bs.append(get_b_by_mean_and_coev(ro, coev_fix))

    # Параметры распределения обслуживания для ИМ СМО:
    params = []
    for j in range(k):
        params.append(rd.Gamma.get_mu_alpha([bs[j][0], bs[j][1]]))

    if is_np:
        smo = SmoImPrty(1, k, "NP")  # создаем ИМ
        v_ch = prty_calc.get_w_np(ls, bs)  # расчет численно и получение результатов
    else:
        smo = SmoImPrty(1, k, "PR")  # создаем ИМ
        v_ch = prty_calc.calc_pr1(ls, bs)['v']  # расчет численно и получение результатов

    # Параметры входных потоков для ИМ СМО:
    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': ls[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    smo.set_sources(sources)
    smo.set_servers(servers_params)

    # Запуск ИМ
    smo.run(num_of_jobs)

    # Получаем результаты ИМ
    v_im = smo.v

    for j in range(k):
        vs_im[j].append(v_im[j][0])
        vs_ch[j].append(v_ch[j][0])

    print_results(v_im, v_ch, k, ro, coev_fix, is_np)

# строим график от ro
fig, ax = plt.subplots()

for i in range(k_fix):
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

# !!! Заготовка для построения зависимостей от коэффициента вариации времени обслуживания

# массивы для накопления средних времен пребывания в системе для численного метода и ИМ

vs_ch = []
vs_im = []

for j in range(k_fix):
    vs_ch.append([])
    vs_im.append([])

# задаем массив коээфициентов вариации
coevs = np.linspace(0.3, 3.0, 12)

for coev in coevs:
    # Вставить код, аналогичный циклу по ro, заменив по смыслу ro на ro_fix, coev_fix - на coev
    pass

# !!! Расскомментировать после реализации цикла вверху
# fig, ax = plt.subplots()
#
# for i in range(k_fix):
#     ax.plot(coevs, vs_im[i], label="ИМ класс {0:d}".format(i + 1))
#     ax.plot(coevs, vs_ch[i], label="Числ класс {0:d}".format(i + 1))
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
