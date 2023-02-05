import random
import matplotlib.pyplot as plt
import numpy as np

def l_i(i, x, x_nodes):
    x_nodes_i = []
    for j in range(len(x_nodes)):
        if(j!=i):
            x_nodes_i.append(x_nodes[j])
    return np.prod((x-x_nodes_i)/(x_nodes[i]-x_nodes_i))

def L(x, x_nodes, y_nodes):
    return np.sum(y_nodes*[l_i(i,x,x_nodes) for i in range(x_nodes.size)])

def plot_data(x_node, f, k, flag):
    y_node = f(x_node)
    fig, ax = plt.subplots(1, 1, figsize = (6, 5))
    plt.gcf().canvas.set_window_title(f'base_{flag}_{k}')
    #определение точек f(x)
    x_plt = np.linspace(-1,1,200)
    ax.plot(x_node, f(x_node), 'ro', markersize = 2)
    ax.plot(x_plt, f(x_plt), 'grey', label = r'$f(x) = \frac{1}{1+25x^2}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    #определение точек интерполянта ~f(x)
    ax.plot(x_plt, [L(i,x_node,y_node) for i in x_plt], label = r'$\tilde{f}(x)$')
    ax.legend()
    ax.grid()

    plt.savefig(f'base_{flag}_{k}.png', dpi = 600)
    plt.close()


def plot_data_res(x_u,x_ch, f, k, flag):
    y_u = f(x_u)
    y_ch = f(x_ch)
    fig, ax = plt.subplots(1, 1, figsize = (6, 5))
    plt.gcf().canvas.set_window_title(f'base_{flag}_{k}')
    #определение точек f(x)
    x_plt = np.linspace(-1,1,200)
    ax.plot(x_u, f(x_u), 'ro', markersize = 2)
    ax.plot(x_plt, f(x_plt), 'grey', label = r'$f(x) = \frac{1}{1+25x^2}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    #определение точек интерполянта ~f(x)
    ax.plot(x_plt, [L(i,x_u,y_u) for i in x_plt], label = r'$\tilde{f}(x)_{uniform}$')
    ax.plot(x_plt, [L(i, x_ch, y_ch) for i in x_plt], label=r'$\tilde{f}(x)_{chebishev}$', color = 'green')
    ax.plot(x_ch, f(x_ch), 'o', color = 'green', markersize=5)
    ax.legend()
    ax.grid()
    plt.savefig(f'base_{flag}_{k}.png', dpi = 600)
    plt.close()

def uniform():
    f = lambda x: 1. / (1. + 25 * x ** 2)
    for n in range(5, 24, 3):
        x = np.linspace(-1,1,n)
        plot_data(x,f, n, 'u')


def chebishev():
    f = lambda x: 1. / (1. + 25 * x ** 2)
    for n in range(5, 24, 3):
        x = np.array([np.cos((2*i-1)/(2*n)*np.pi) for i in range(1, n+1)])
        plot_data(x,f, n, 'c')

def uniform_chebishev():
    f = lambda x: 1. / (1. + 25 * x ** 2)
    for n in range(5, 24, 3):
        x_u = np.linspace(-1,1,n)
        x_ch = np.array([np.cos((2*i-1)/(2*n)*np.pi) for i in range(1, n+1)])
        plot_data_res(x_u, x_ch,f, n, 'uc')