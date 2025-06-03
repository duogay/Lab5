import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as w
from scipy.signal import butter, filtfilt

default_amplitude = 1.0
default_frequency = 1.0
default_phi = 0.0

default_noise_mean = 0.0
default_noise_var = 0.1

default_filter_order = 4
default_cutoff = 5.0

t = np.linspace(0, 2, 2000)

def generate_y(amplitude, frequency, phi):
    w = 2*np.pi*frequency
    return amplitude * np.sin(w * t + phi)

def generate_noise(noise_mean, noise_var):
    return np.random.normal(noise_mean, np.sqrt(noise_var), len(t))

def filter(y, order, cutoff):
    sampling_rate = 1 / (t[1] - t[0])
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, y)

current_y = generate_y(default_amplitude, default_frequency, default_phi)
current_noise = generate_noise(default_noise_mean, default_noise_var)
current_filter = filter(current_y + current_noise, default_filter_order, default_cutoff)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.55, top=1)

[noise] = ax.plot(t, current_y + current_noise)
[graph] = ax.plot(t, current_y)
[filtered] = ax.plot(t, current_filter)
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.grid(True)

ax_amp = plt.axes([0.15, 0.32, 0.7, 0.03])
ax_freq = plt.axes([0.15, 0.27, 0.7, 0.03])
ax_phi = plt.axes([0.15, 0.22, 0.7, 0.03])

ax_noise_mean = plt.axes([0.15, 0.17, 0.7, 0.03])
ax_noise_var = plt.axes([0.15, 0.12, 0.7, 0.03])

ax_reset = plt.axes([0.05, 0.4, 0.1, 0.04])
ax_show_noise = plt.axes([0.4, 0.4, 0.1, 0.04])
ax_show_filter = plt.axes([0.8, 0.4, 0.1, 0.04])

ax_filter_order = plt.axes([0.15, 0.07, 0.7, 0.03])
ax_filter_cutoff = plt.axes([0.15, 0.02, 0.7, 0.03])

slider_amp = w.Slider(ax_amp, 'Амплітуда', 0.1, 5.0, valinit=default_amplitude)
slider_freq = w.Slider(ax_freq, 'Частота', 0.1, 5.0, valinit=default_frequency)
slider_phi = w.Slider(ax_phi, 'Фазовий зсув', -np.pi*2, np.pi*2, valinit=default_phi)

slider_noise_mean = w.Slider(ax_noise_mean, 'Середнє значення', -1.0, 1.0, valinit=default_noise_mean)
slider_noise_var = w.Slider(ax_noise_var, 'Дисперсія шуму', 0.0, 1.0, valinit=default_noise_var)

slider_filter_order = w.Slider(ax_filter_order, 'Порядок фільтра', 1, 9, valinit=default_filter_order)
slider_filter_cutoff = w.Slider(ax_filter_cutoff, 'Частота зрізу', 1, 9, valinit=default_cutoff)

button_reset = w.Button(ax_reset, 'Скинути')
check_show_noise = w.CheckButtons(ax_show_noise, ['Шум'], [True])
check_show_filter = w.CheckButtons(ax_show_filter, ['Фільтр'], [True])


def update_y(event = None):
    global current_y
    global current_filter

    amp = slider_amp.val
    freq = slider_freq.val
    phi = slider_phi.val

    order = int(slider_filter_order.val)
    cutoff = int(slider_filter_cutoff.val)

    current_y = generate_y(amp, freq, phi)
    current_filter = filter(current_y + current_noise, order, cutoff)
    
    graph.set_ydata(current_y)
    noise.set_ydata(current_y + current_noise)
    filtered.set_ydata(current_filter)

    fig.canvas.draw_idle()

def update_noise(event = None):
    global current_noise
    global current_filter

    noise_mean = slider_noise_mean.val
    noise_var = slider_noise_var.val
    
    order = int(slider_filter_order.val)
    cutoff = int(slider_filter_cutoff.val)

    current_noise = generate_noise(noise_mean, noise_var)
    current_filter = filter(current_y + current_noise, order, cutoff)

    noise.set_ydata(current_y + current_noise)
    filtered.set_ydata(current_filter)
    
    fig.canvas.draw_idle()

def update_filter(event = None):
    global current_filter

    order = int(slider_filter_order.val)
    cutoff = int(slider_filter_cutoff.val)

    current_filter = filter(current_y + current_noise, order, cutoff)

    filtered.set_ydata(current_filter)
    
    fig.canvas.draw_idle()

def reset(event = None):
    slider_amp.reset()
    slider_freq.reset()
    slider_phi.reset()
    slider_noise_mean.reset()
    slider_noise_var.reset()

    if not check_show_noise.get_status()[0]:
        check_show_noise.set_active(0)
        noise.set_visible(True)

    if not check_show_filter.get_status()[0]:
        check_show_filter.set_active(0)
        filtered.set_visible(True)


def toggle_noise(event = None): 
    noise.set_visible(check_show_noise.get_status()[0])
    fig.canvas.draw_idle()

def toggle_filter(event = None):
    filtered.set_visible(check_show_filter.get_status()[0])
    fig.canvas.draw_idle()

slider_amp.on_changed(update_y)
slider_freq.on_changed(update_y)
slider_phi.on_changed(update_y)

slider_noise_mean.on_changed(update_noise)
slider_noise_var.on_changed(update_noise)

slider_filter_cutoff.on_changed(update_filter)
slider_filter_order.on_changed(update_filter)

button_reset.on_clicked(reset)

check_show_noise.on_clicked(toggle_noise)
check_show_filter.on_clicked(toggle_filter)

plt.show()