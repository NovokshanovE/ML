import random
import matplotlib.pyplot as plt
import numpy as np
from base_1 import *
class function:
    def __init__(self, seed):
        n = random.randint(7, 15)
        m = random.randint(7, 15)
        j = [random.randint(0, 1000) / 1000 for i in range(m)]
        k = [random.randint(0, 1000) / 1000 for i in range(n)]
        self.n = n
        self.m = m
        self.j = j
        self.k = k
    def read(self):
        return self.n, self.m, self.j, self.k
    def function_value_from_x(self, x):
        a = []
        b = []
        for i in range(len(self.j)):
            a.append(self.j[i] * x ** i)
        for i in range(len(self.k)):
            b.append(self.k[i] * x ** i)
        A = np.array(a)
        B = np.array(b)
        return np.sum(A) / (1 + (np.sum(B)))
#def generation():
#    n = random.randint(7, 15)
#    m = random.randint(7, 15)
#    j = [random.randint(0, 1000) / 1000 for i in range(0, m + 1)]
#    k = [random.randint(0, 1000) / 1000 for i in range(0, n + 1)]
#    return n, m, j, k
#def func_random(x,m, n, j, k):
#    a = []
#    b = []
#    for i in range(len(j)):
#        a.append(j[i]*x**i)
#    for i in range(len(k)):
#        b.append(k[i]*x**i)
#    A = np.array(a)
#    B = np.array(b)
#    return np.sum(A)/(1+(np.sum(B)))
def tsk1_2(flag):
    functions = [function(random.seed(i+3)) for i in range(100)]
    N = 14
    funk_res = []
    for l in range(4):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.gcf().canvas.set_window_title(f'advanced_tsk1_{l+1}')
        i = 25*l
        funk_res.append(functions[i])
        # определение точек f(x)
        x_plt = np.linspace(-1, 1, 100)
        y_plt = np.array([functions[i].function_value_from_x(x) for x in x_plt])
        x_node = np.linspace(-1,1,N)#np.array([x_plt[x] for x in range(1, len(x_plt)+1,100//N)])
        y_node = np.array([functions[i].function_value_from_x(x) for x in x_node])
        x_chebishev = np.array([np.cos((2 * i - 1) / (2 * N) * np.pi) for i in range(1, N + 1)])
        y_chebishev = np.array([functions[i].function_value_from_x(x) for x in x_chebishev])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        if(flag):
            ax.plot(x_plt, [L(x, np.array(x_node), np.array(y_node)) for x in np.array(x_plt)], 'blue', label = r'$\tilde{f}(x)_{uniform}$')
            ax.plot(x_node, y_node, 'ro', markersize=3)
            ax.plot(x_plt, y_plt, 'grey', label = 'график')
            ax.grid()
            ax.legend()
            ax.plot(x_plt, [L(x, np.array(x_chebishev), np.array(y_chebishev)) for x in np.array(x_plt)], 'green', label = r'$\tilde{f}(x)_{advenced}$')
            ax.plot(x_chebishev, y_chebishev, 'ro', markersize=3)
            ax.plot(x_plt, y_plt, 'grey', label = 'график')
            ax.grid()
            ax.legend()
        plt.savefig(f'advanced_tsk1_{l+1}.png', dpi=600)
    if(flag):
        plt.show()
    return funk_res
def tsk3(functions):
    N = 30
    for l in range(4):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plt.gcf().canvas.set_window_title(f'advanced_tsk2_{l + 1}')
        x_plt = np.linspace(-1, 1, 200)
        y_plt = np.array([functions[l].function_value_from_x(x) for x in x_plt])
        x_node = np.linspace(-1,1,N)#np.array([x_plt[x] for x in range(0, len(x_plt), 100 // N)])
        y_node = np.array([functions[l].function_value_from_x(x) for x in x_node])
        x_chebishev = np.array([np.cos((2 * i - 1) / (2 * N) * np.pi) for i in range(1, N + 1)])
        y_chebishev = np.array([functions[l].function_value_from_x(x) for x in x_chebishev])

        ax.plot(x_plt, [L(x, np.array(x_node), np.array(y_node)) for x in np.array(x_plt)], 'blue')
        ax.plot(x_plt, [L(x, np.array(x_chebishev), np.array(y_chebishev)) for x in np.array(x_plt)], 'green')
        ax.plot(x_node, y_node, 'ro', markersize=5, label='узлы')
        ax.plot(x_plt, y_plt, 'grey', label='график')
        ax.grid()
        #ax[0].legend()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        plt.savefig(f'advanced_tsk2_{l + 1}_1.png', dpi=600)
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        x_ro = np.array([N_i for N_i in range(1,31)])
        y_ro_1 = []
        y_ro_2 = []
        for N_i in x_ro:
            x_node = np.linspace(-1,1,N_i)#np.array([x_plt[x] for x in range(0, len(x_plt), 100 // N_i)])
            #print([x_plt[x] for x in range(0, len(x_plt), 100 // N_i)])
            y_node = np.array([functions[l].function_value_from_x(x) for x in x_node])

            x_chebishev = np.array([np.cos((2 * i - 1) / (2 * N_i) * np.pi) for i in range(1, N_i + 1)])
            y_chebishev = np.array([functions[l].function_value_from_x(x) for x in x_chebishev])

            y = max([abs(L(x, np.array(x_node), np.array(y_node))-functions[l].function_value_from_x(x)) for x in np.array(x_plt)])
            y_ro_1.append(y)
            y_ch = max([abs(L(x, np.array(x_chebishev), np.array(y_chebishev))-functions[l].function_value_from_x(x)) for x in np.array(x_plt)])
            y_ro_2.append(y_ch)
        ax.semilogy(x_ro, y_ro_1, 'blue', label=r'$||g(x)||_{\infty}uniform$')
        ax.semilogy(x_ro, y_ro_2, 'green', label=r'$||g(x)||_{\infty}chebishev$')
        ax.legend()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        plt.savefig(f'advanced_tsk2_{l+1}_2.png', dpi=600)
        #plt.close()
    plt.show()
    return 0