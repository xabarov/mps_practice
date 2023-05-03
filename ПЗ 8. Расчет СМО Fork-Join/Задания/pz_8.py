import mg1_calc
import rand_destribution as rd
from fj_im import SmoFJ
import matplotlib.pyplot as plt
import math

def dfr_Gamma_Mult(params, t, n):
    """
    Вычисление ДФР
    """
    res = 1.0
    for i in range(n):
        res *= rd.Gamma.get_cdf(*params, t)
    return 1.0 - res


def getMaxMoments(n, b, num=3):
    """
    Расчет начальных моментов максимума СВ.
    :param n: число одинаково распределенных СВ
    :param b: начальные моменты СВ
    :param num: число нач. моментов СВ
    :return: начальные моменты максимума СВ.
    """
    f = [0] * num

    a_big = [1.37793470540E-1, 7.29454549503E-1, 1.808342901740E0,
             3.401433697855E0, 5.552496140064E0, 8.330152746764E0,
             1.1843785837900E1, 1.6279257831378E1, 2.1996585811981E1,
             2.9920697012274E1]
    g = [3.08441115765E-1, 4.01119929155E-1, 2.18068287612E-1,
         6.20874560987E-2, 9.50151697518E-3, 7.53008388588E-4,
         2.82592334960E-5, 4.24931398496E-7, 1.83956482398E-9,
         9.91182721961E-13]

    params = rd.Gamma.get_mu_alpha(b)

    for j in range(10):
        p = g[j] * dfr_Gamma_Mult(params, a_big[j], n) * math.exp(a_big[j])
        f[0] += p
        for i in range(1, num):
            p = p * a_big[j]
            f[i] += p

    for i in range(num - 1):
        f[i + 1] *= (i + 2)
    return f


def calc_error(im, ch):
    return 100 * (im - ch) / im


def calc_approximation(l, mu, n):
    """
    Напишите функцию расчета среднего времени пребывания заявок
    в СМО Fork-Join с помощью аппроксимации, в зависимости от вашего варианта:
        - Nelson R. и Tantawi A.N.
        - Varki E., Merchant A. и Chen H.
        - Varma S. и Makowski A.M.

    :param l: интенсивность вх потока заявок
    :param mu: интенсивность обслуживания заявок в канале
    :param n: число каналов обслуживания

    :return: v1 - среднее время пребывания заявок в СМО
    """
    pass


# Задаем список значений числа каналов обслуживания в диапазоне от 2 до 10 включительно:
ns = [n for n in range(2, 11)]

# интенсивность вх потока заявок
l = 1.0

# Число заявок, требуемых к обслуживанию в СМО Split-Join и Fork-Join
n_jobs = 3000

# Коэффициент загрузки при n=1.
ro_start = 0.9

# Списки для накопления средних времен пребывания заявок в СМО Split-Join для ИМ и рассчитанных численно
v1_sj_im = []
v1_sj_ch = []

# Списки для накопления средних времен пребывания заявок в СМО Fork-Join для ИМ
# и рассчитанных численно (на основе аппрокимации согласно вашего варианта)

v1_fj_im = []
v1_fj_ch = []

# Относительная ошибка аппроксимации для СМО Fork-Join
errors_fj = []

# Ускорение (уменьшение среднего времени пребывания заявок в СМО) при увеличении каналов обслуживания
fj_speed_ups = []
sj_speed_ups = []

# Вычисляем время обработки без ускорений
# ro = l*b1/n
# -> b1 = ro*n/l = ro -> mu = 1.0/b1 = 1.0/ro
mu_one = 1 / ro_start
b_one_channel = rd.Exp_dist.calc_theory_moments(mu_one, 3)
v_base = mg1_calc.get_v(l, b_one_channel)[0]

for n in ns:
    # 1. ИМ Split-Join

    # Задаем ИМ. Создаем экземпляр класса ИМ:
    smo = SmoFJ(n, n, True)
    # Задаем источник заявок (входящий поток заявок)
    smo.set_sources(l, 'M')

    # Пересчитываем интенсивность обслуживания заявок
    mu = mu_one * n

    # Задаем каналы обслуживания ИМ
    smo.set_servers(mu, 'M')

    # Запускаем ИМ Split-Join и добавляем результат в соответствующий список
    smo.run(n_jobs)
    v1_sj_im.append(smo.v[0])

    # 2. Численный расчет СМО Split-Join через распределение максимума СВ
    b_one_channel = rd.Exp_dist.calc_theory_moments(mu, 3)
    b_max = getMaxMoments(n, b_one_channel)
    v_ch = mg1_calc.get_v(l, b_max)[0]
    v1_sj_ch.append(v_ch)

    # Ускорение - как отношение одноканального варианта СМО
    # ср времени пребвания при текущем значении n
    sj_speed_ups.append(v_base / smo.v[0])

    # 3. Fork-Join

    # Задаем ИМ Fork-Join аналогично ИМ Split-Join,
    # указываем третий параметр is_SJ = False
    smo = SmoFJ(n, n, False)
    smo.set_sources(l, 'M')
    smo.set_servers(mu, 'M')

    # Запускаем ИМ и сохраняем результаты
    smo.run(n_jobs)
    v1_fj_im.append(smo.v[0])
    fj_speed_ups.append(v_base / smo.v[0])

    # !!! Рассчитываем среднее время пребывания в Fork-Join
    # на основе аппроксимации согласно вашего варианта
    v_ch = calc_approximation(l, mu, n)

    v1_fj_ch.append(v_ch)
    errors_fj.append(calc_error(smo.v[0], v_ch))

# Построение графиков:
fig, ax = plt.subplots()

ax.plot(ns, v1_sj_im, label="SJ ИМ")
ax.plot(ns, v1_sj_ch, label="SJ Числ")
ax.plot(ns, v1_fj_im, label="FJ ИМ")
ax.plot(ns, v1_fj_ch, label=f"FJ Числ")

ax.set_xlabel('n')
ax.set_ylabel(r"$\upsilon_{1}$")
plt.legend()
plt.show()

fig, ax = plt.subplots()

ax.plot(ns, errors_fj)

ax.set_xlabel('n')
ax.set_ylabel("error, %")
plt.legend()
plt.show()

fig, ax = plt.subplots()

ax.plot(ns, sj_speed_ups, label="SJ")
ax.plot(ns, fj_speed_ups, label="FJ")

ax.set_xlabel('n')
ax.set_ylabel("ускорение, раз")
plt.legend()
plt.show()
