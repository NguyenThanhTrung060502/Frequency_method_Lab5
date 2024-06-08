import numpy as np 
import matplotlib.pyplot as plt


# Hàm chữ nhật
def rectangle(t, a, b):
    if abs(t) <= b:
        return a
    return 0

# Vẽ hàm số 
def plot_func(t, func, color, title, legend, labels):
    ymin = -0.2
    ymax = 1.2
    plt.figure(figsize=(8, 5))
    plt.ylim(ymin, ymax) 
    # plt.xlim(-3, 3)
    plt.plot(t, func.real, color=f'{color}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

# Vẽ ảnh Fourier của hàm số 
def plot_image(v, func, colors, title, legend, labels, xlim):
    plt.figure(figsize=(8, 5)) 
    plt.plot(v, func.real, color=colors[0])
    plt.plot(v, func.imag, color=colors[1])
    plt.xlim(-xlim, xlim)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)   

# So sánh hàm ban đầu và hàm được phục hồi.
def cmp_func(t, func, colors, title, legend, labels, linestyles, xlim):
    plt.figure(figsize=(8, 5))
    plt.plot(t, func[0], color=colors[0], linestyle=linestyles[0])
    plt.plot(t, func[1], color=colors[1], linestyle=linestyles[1])
    plt.xlim(-xlim, xlim)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5) 

# Phương pháp DFT 
def dft(step): 
    # Прямоугольная функция 
    f = np.vectorize(lambda t :rectangle(t, 1, 0.5), otypes=[np.complex_])
    t = np.linspace(-5, 5, step)
    f = f(t)

    # calc DFT 
    v = np.fft.fftshift(np.fft.fftfreq(step, 10 / step)) 
    image_dft = np.fft.fftshift(np.fft.fft(f, norm='ortho')) 
    plot_image(v, np.sinc(v), colors=['green', 'red'], title='Истинный Фурье-образ функции Π(t)', legend=['Real', 'Imag'], labels=['v', r'$\hat{П}(v)$'], xlim=15)
    plot_image(v, image_dft, colors=['green', 'red'], title='Фурье-образ функции Π(t) с помощью дискретного преобразования Фурье', legend=['Real', 'Imag'], labels=['v', r'$\hat{П}(v)$'], xlim=15)

    # Обратный DFT 
    f_restored = np.fft.ifft(np.fft.ifftshift(image_dft), norm='ortho') 
    plot_func(t, f_restored, color='black', title='Восстановленная функция', legend=['Восстановленная функция'], labels=['t', 'П(t)'])
    cmp_func(t, [f, f_restored], colors=['black', 'cyan'], title='Сравнение', legend=['Исходная', 'Восстановленная'], labels=['t', 'П(t)'], linestyles=['-', '--'], xlim=15)


    # дискретный образ
    dt = t[1] - t[0]
    dft_cont_image = image_dft * dt * np.exp(-1j * v * 2 * np.pi * t[0]) * np.sqrt(step)
    plot_image(v, dft_cont_image, colors=['green', 'red'], title='DFT образ', legend=['Real', 'Imag'], labels=['v', r'$\hat{П}(v)$'], xlim=15)
    cmp_func(v, [np.sinc(v), dft_cont_image], colors=['black', 'cyan'], title='Сравнение', legend=['True image', 'DFT image'], labels=['v', r'$\hat{П}(v)$'], linestyles=['-', '--'], xlim=15)

    # восстановленная функция из дискретного образa
    dv = v[1] - v[0]
    f_restored_cont = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(dft_cont_image), norm='ortho') * np.sqrt(step) * dv)
    # plot_func(t, f_restored_cont, color='green', title='Восстановленная функция', legend=['П(t)'], labels=['t', 'П(t)'])
    cmp_func(t, [f, f_restored_cont], colors=['black', 'cyan'], title='Сравнение', legend=['Исходная', 'Восстановленная'], labels=['t', 'П(t)'], linestyles=['-', '--'], xlim=2.5)

dft(200)
dft(9998)
# plt.show()