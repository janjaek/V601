import matplotlib as mpl

mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy
from uncertainties import ufloat

data = np.genfromtxt("content/energie20.txt", unpack=True)

A1 = ufloat(2.27, 0.03)

data2 = np.zeros(data[0].size -1)
for i in range(data[1].size-1):
	data2[i] = data[1][i] - data[1][i+1]
data2 = unumpy.uarray(data2, data2/A1.n*A1.s)

plt.errorbar(data[0][:-1], unumpy.nominal_values(data2), xerr=unumpy.std_devs(data2), elinewidth=0.7, capthick=0.7, capsize=3, fmt=".", color="xkcd:blue", label=r"Messwerte für $\SI{20}{\degreeCelsius}$")
plt.xlabel(r"$U_A/\si{\volt}$")
plt.ylabel(r"$\propto I_A(U_A)-I_A(U_A+\Delta U_A)$")
plt.grid(which="both")
plt.legend()
plt.tight_layout()
plt.savefig("build/energie20.pdf")
plt.clf()
print(data2[-4], data[0][-5])

data = np.genfromtxt("content/energie140.txt", unpack=True)

data2 = np.zeros(data[0].size -1)
for i in range(data[1].size-1):
	data2[i] = data[1][i] - data[1][i+1]

plt.plot(data[0][:-1], data2, ".", color="xkcd:blue", label=r"Messwerte für $\SI{140}{\degreeCelsius}$")
plt.xlabel(r"$U_A/\si{\volt}$")
plt.ylabel(r"$\propto I_A(U_A)-I_A(U_A+\Delta U_A)$")
plt.grid(which="both")
plt.legend()
plt.tight_layout()
plt.savefig("build/energie140.pdf")
plt.clf()


def f(x, m, n):
	return m*x + n


data = np.genfromtxt("content/ion.txt", unpack=True)

params, covar = curve_fit(f, data[0], data[1])
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print("Parameter m und n für den lin. Fit: ")
print(uparams)

print(-uparams[1]/uparams[0])
lin = np.linspace(-params[1]/params[0] -1, data[0][-1], 1000)

A2 = ufloat(0.46,0.01)
data2 = unumpy.uarray(data[0], data[0]/A2.n*A2.s)

plt.plot(lin, f(lin, *params), color="xkcd:orange", label="linearer Fit")
plt.errorbar(unumpy.nominal_values(data2), data[1], fmt=".", xerr=unumpy.std_devs(data2), elinewidth=0.7, capthick=0.7, capsize=3, color="xkcd:blue", label="Messwerte")
plt.errorbar([-params[1]/params[0]], [0], xerr=unumpy.std_devs(-uparams[1]/uparams[0]), elinewidth=0.7, capthick=0.7, capsize=3, fmt="x", color="xkcd:green", label="Schnittpunkt")
plt.xlabel(r"$U_B/\si{\volt}$")
plt.ylabel(r"$\propto I_A(U_B)$")
plt.grid(which="both")
plt.legend()
plt.tight_layout()
plt.savefig("build/ion.pdf")
plt.clf()
