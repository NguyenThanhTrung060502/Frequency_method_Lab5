import numpy as np 
import matplotlib.pyplot as plt 

pi= np.pi

# Hàm trapz()
def trapz_function(t, f, g):
    dt = t[1] - t[0]
    return np.trapz(f*g, dx=dt)

# Vẽ hàm số 
def plot_func(t, func, color, title, legend, labels, xlim):
    plt.figure(figsize=(9, 6))
    plt.plot(t, func.real, color=f'{color}')
    plt.xlim(-xlim, xlim)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

# Vẽ ảnh Fourier của hàm số 
def plot_image(v, func, colors, title, legend, labels):
    plt.figure(figsize=(9, 6)) 
    plt.plot(v, func.real, color=colors[0], linewidth=1.25)
    plt.plot(v, func.imag, color=colors[1], linewidth=0.5)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)   

# So sánh hàm ban đầu và hàm được phục hồi.
def cmp_func(t, func, colors, title, legend, labels, linestyles):
    plt.figure(figsize=(9, 6))
    plt.plot(t, func[0], color=colors[0], linestyle=linestyles[0])
    plt.plot(t, func[1], color=colors[1], linestyle=linestyles[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5) 

# So sánh 2 
def cmp_func_diff(t, func, title, colors, legend, labels, linestyles, xlim):
    plt.figure(figsize=(9, 6)) 
    plt.plot(t[0], func[0], color=colors[0], linestyle=linestyles[0])
    plt.plot(t[1], func[1], color=colors[1], linestyle=linestyles[1])
    plt.xlim(-xlim, xlim)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)


# Фурье-образ 
get_fourier_image = lambda X, W, func: np.array([trapz_function(X, func, (lambda t: np.exp(-1j*2*pi*w*t))(X)) for w in W])  

    
def interpolation(func, dt, B):
    t_cont = np.linspace(-55, 55, 5555)
    plot_func(t_cont, func(t_cont),color='black', title='Исходная функция',legend=['f(t)'], labels=['t', 'f(t)'], xlim=5)

    t_sampled = np.arange(-55, 55, dt)
    f_sampled = func(t_sampled)
    plot_func(t_sampled, f_sampled, color='red', title='Сэмплированная функция', legend=['Сэмплированная'], labels=['t', 'f(t)'], xlim=5)
    cmp_func_diff([t_cont, t_sampled], [func(t_cont), f_sampled], colors=['black', 'red'], title='Сравнение',  legend=['Исходная', 'Сэмплированная'], labels=['t', 'f(t)'], linestyles=['-', '--'], xlim=9)

    # Фурбе-образ 
    V = np.linspace(-22, 22, 2222)
    image = get_fourier_image(t_sampled, V, f_sampled)
    plot_image(V, image, colors=['green', 'red'], title='Фурбе-образ исходной функции', legend=['Real', 'Imag'], labels=['v', r'$\hat{f}(v)$'])

    interpolation 
    f_interp = np.vectorize(lambda t: np.sum([func(n) * np.sinc(2 * B * (t - n)) for n in t_sampled]))
    t_interp = np.linspace(-10, 10, 1000)
    plot_func(t_interp, f_interp(t_interp), color='black', title='Интерполированная функция', legend=['Интерполированная'], labels=['t', 'f(t)'], xlim=9)
    cmp_func_diff([t_cont, t_interp], [func(t_cont), f_interp(t_interp)], colors=['black', 'magenta'], title='Сравнение',  legend=['Исходная', 'Интерполированная'], labels=['t', 'f(t)'], linestyles=['-', '--'], xlim=9)

    restored_image = get_fourier_image(t_interp, V, f_interp(t_interp))
    plot_image(V, restored_image, colors=['green', 'red'], title='Фурбе-образ восстановленного функции', legend=['Real', 'Imag'], labels=['v', r'$\hat{f}(v)$'])



sin_func = lambda a1,a2,w1,w2,p1,p2: np.vectorize(lambda x: a1 * np.sin(w1 * x + p1) + a2 * np.sin(w2 * x - p2))
y = sin_func(3, 5, pi, 2*pi, 5*pi, 4*pi)
# interpolation(y, dt=1/9, B=4.5) 
# interpolation(y, dt=1/4, B=5)
# interpolation(y, dt=1/10, B=5)


sinc = lambda b: np.vectorize (lambda x: np.sinc(b * x))
sinc = sinc(5)
# interpolation(sinc, dt=1/6, B=3)
# interpolation(sinc, dt=1/5, B=10)

# plt.show()
