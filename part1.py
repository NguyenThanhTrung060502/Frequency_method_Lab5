import numpy as np 
import matplotlib.pyplot as plt 

pi = np.pi

# Hàm chữ nhật
def rectangle(t, a, b):
    if abs(t) <= b:
        return a
    return 0

# Hàm trapz()
def trapz_function(t, f, g):
    dt = t[1] - t[0]
    return np.trapz(f*g, dx=dt)

# Vẽ hàm số 
def plot_func(t, func, color, title, legend, labels):
    ymin = -0.2
    ymax = 1.2
    plt.figure(figsize=(8, 5))
    plt.ylim(ymin, ymax) 
    plt.plot(t, func.real, color=f'{color}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

# Vẽ ảnh Fourier của hàm số 
def plot_image(v, func, colors, title, legend, labels):
    plt.figure(figsize=(8, 5)) 
    plt.plot(v, func.real, color=colors[0])
    plt.plot(v, func.imag, color=colors[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)   

# So sánh hàm ban đầu và hàm được phục hồi.
def cmp_func(t, func, colors, title, legend, labels, linestyles):
    plt.figure(figsize=(8, 5))
    plt.plot(t, func[0], color=colors[0], linestyle=linestyles[0])
    plt.plot(t, func[1], color=colors[1], linestyle=linestyles[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5) 


# steps = np.array([1000, 5000, 10000])
# Vs = np.array([20, 50, 100])

step = 10000
vs = 100

t = np.linspace(-2, 2, step)
v = np.linspace(-vs, vs, step)

# Прямоугольная функция Π(t)
f = np.vectorize(lambda t :rectangle(t, 1, 0.5), otypes=[np.complex_])
f_true = f(t)

#  Истинный Фурье-образ функции Π(t) 
true_image = (lambda v: np.sinc(v))(v)

# Численное интегрирование
get_fourier_image = lambda X, W, func: np.array([trapz_function(X, func, (lambda t: np.exp(-1j*2*pi*w*t))(X)) for w in W])
trapz_image = get_fourier_image(t, v, f_true)

# Восстановленная функция Π(t)
get_fourier_function = lambda X, W, func_image: np.array([trapz_function(W, func_image, (lambda t: np.exp(1j*2*pi*x*t))(W)) for x in X])
restored_num = get_fourier_function(t, v, trapz_image)



# Vẽ hàm ban đầu 
plot_func(t, f_true, color='black', title='Прямоугольная функция', legend=['П(t)'], labels=['t', 'П(t)'])

# Vẽ истинный Фурье-образ функции Π(t)
plot_image(v, true_image, colors=['green', 'red'], title='Истинный Фурье-образ функции Π(t)', legend=['Real', 'Imag'], labels=['v', r'$\hatП(v)$'])

# Vẽ hàm được khôi phục lại 
plot_func(t, restored_num, color='gold', title='Восстановленная функция Π(t)', legend=['Π(t)'], labels=['t', 'П(t)'])

# Vẽ Фурье-образ функции Π(t) bằng cách dùng hàm trapz
plot_image(v, trapz_image, colors=['darkcyan', 'orange'], title='Фурье-образ функции Π(t) с помощью функция trapz', legend=['Real', 'Imag'], labels=['v', r'$\hatП(v)$'])

# Vẽ đồ thị để so sánh hàm ban đầu với hàm được phục hồi 
cmp_func(t, func=[f_true, restored_num], colors=['black', 'tomato'], title='Сравните исходную и восстановленную функцию', legend=['Исходная', 'Восстановленная'], labels=['t', 'П(t)'], linestyles=['-', '--'])

# Vẽ đồ thị để so sánh Фурье-образ исходного и численного
cmp_func(v, func=[true_image, trapz_image], colors=['cyan', 'black'], title='Сравнение Фурье-образа исходного и численного', legend=['Исходный', 'Численный'], labels=['v', r'$\hat{П}(v)$'], linestyles=['-', '--'])


# plt.show()