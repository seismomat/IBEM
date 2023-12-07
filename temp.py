# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy.core import Trace, Stream

# Parámetros de las trazas sísmicas
num_traces = 5  # Número de trazas sísmicas
sampling_rate = 100  # Tasa de muestreo en Hz
duration = 10  # Duración en segundos
freq = 1  # Frecuencia de la onda en Hz

# Generar el tiempo para las trazas
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Crear un Stream de ObsPy para almacenar las trazas sísmicas
stream = Stream()

# Generar trazas sísmicas simuladas
for i in range(num_traces):
    # Generar una señal sísmica simple (por ejemplo, una onda sinusoidal)
    signal = np.sin(2 * np.pi * freq * t)

    # Crear un objeto Trace de ObsPy
    trace = Trace(data=signal)
    trace.stats.sampling_rate = sampling_rate
    trace.stats.starttime = 0  # Tiempo de inicio de la traza
    trace.stats.network = "SY"
    trace.stats.station = f"STATION_{i}"  # Nombre de la estación
    trace.stats.channel = "HHZ"  # Canal (vertical)

    # Agregar la traza al Stream
    stream.append(trace)

# Graficar las trazas sísmicas
plt.figure(figsize=(10, 6))
for i, trace in enumerate(stream):
    plt.plot(trace.times(), trace.data + i * 2, label=trace.stats.station)

plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Trazas Sísmicas Simuladas')
plt.legend()
plt.grid(True)
plt.show()
