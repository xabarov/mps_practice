import mg1_calc
import fj_calc
import rand_destribution as rd
from fj_im import SmoFJ
import matplotlib.pyplot as plt


def calc_error(im, ch):
    return 100 * (im - ch) / im


ns = [n for n in range(2, 11)]
l = 1.0
n_jobs = 30000
ro_start = 0.9

v1_sj_im = []
v1_sj_ch = []

v1_fj_im = []
v1_fj_ch = []
errors_fj = []

fj_speed_ups = []
sj_speed_ups = []

fj_methods = {
    # "Varma": fj_calc.get_v1_fj_varma,
    # "Nelson": fj_calc.get_v1_fj_nelson_tantawi,
    "Varki": fj_calc.get_v1_fj_varki_merchant
}

for m in fj_methods:
    v1_fj_ch.append([])
    errors_fj.append([])

# Вычисляем время обработки без ускорений
mu = 1 / ro_start
b_one_channel = rd.Exp_dist.calc_theory_moments(mu, 3)
v_base = mg1_calc.get_v(l, b_one_channel)[0]


for n in ns:
    # Split-Join
    smo = SmoFJ(n, n, True)
    smo.set_sources(l, 'M')

    mu = n / ro_start

    b_one_channel = rd.Exp_dist.calc_theory_moments(mu, 2)

    smo.set_servers(mu, 'M')

    smo.run(n_jobs)
    v1_sj_im.append(smo.v[0])

    b_max = fj_calc.getMaxMoments(n, b_one_channel)
    v_ch = mg1_calc.get_v(l, b_max)[0]
    v1_sj_ch.append(v_ch)
    sj_speed_ups.append(v_base/smo.v[0])

    # Fork-Join

    smo = SmoFJ(n, n, False)
    smo.set_sources(l, 'M')

    smo.set_servers(mu, 'M')
    smo.run(n_jobs)
    v1_fj_im.append(smo.v[0])

    fj_speed_ups.append(v_base / smo.v[0])

    for j, m in enumerate(fj_methods):
        v_ch = fj_methods[m](l, mu, n)
        v1_fj_ch[j].append(v_ch)
        errors_fj[j].append(calc_error(smo.v[0], v_ch))


fig, ax = plt.subplots()

ax.plot(ns, v1_sj_im, label="SJ ИМ")
ax.plot(ns, v1_sj_ch, label="SJ Числ")
ax.plot(ns, v1_fj_im, label="FJ ИМ")

for j, m in enumerate(fj_methods):
    ax.plot(ns, v1_fj_ch[j], label=f"FJ Числ {m}")

ax.set_xlabel('n')
ax.set_ylabel(r"$\upsilon_{1}$")
plt.legend()
plt.show()

fig, ax = plt.subplots()

for j, m in enumerate(fj_methods):
    ax.plot(ns, errors_fj[j], label=f"{m}")

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
