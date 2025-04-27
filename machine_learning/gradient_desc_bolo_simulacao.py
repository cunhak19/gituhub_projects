
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- Parte 1: Superfície 3D com sabor como terceira variável ----------

# Simular dados
tempo = np.linspace(10, 40, 50)
temperatura = np.linspace(150, 250, 50)
sabor = np.array([0, 1, 2])
T, Temp = np.meshgrid(tempo, temperatura)

# Função da nota do bolo
def nota_bolo(t, temp, sabor):
    return 10 - ((t - 25)**2 / 20 + (temp - 200)**2 / 100 + sabor * 1.5)

# Gráfico 3D para cada sabor
fig = plt.figure(figsize=(18, 5))
for i, s in enumerate(sabor):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    nota = nota_bolo(T, Temp, s)
    surf = ax.plot_surface(T, Temp, nota, cmap='viridis', edgecolor='none')
    ax.set_title(f"Sabor: {['Chocolate', 'Baunilha', 'Morango'][s]}")
    ax.set_xlabel("Tempo (min)")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_zlabel("Nota do Bolo")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()

# ---------- Parte 2: Gradient Descent no espaço normalizado (z-score) ----------

# Função z-score
def z_score(x):
    return (x - np.mean(x)) / np.std(x)

# Normalizar dados
tempo_z = z_score(tempo)
temperatura_z = z_score(temperatura)
sabor_z = z_score(sabor)
Tz, Tempz = np.meshgrid(tempo_z, temperatura_z)

# Função de nota normalizada
def nota_bolo_z(t, temp, sabor):
    return 10 - ((t)**2 + (temp)**2 + sabor * 1.5)

# Simular caminho do Gradient Descent
tempo_ini = -2.0
temp_ini = 2.0
sabor_ini = sabor_z[-1]  # sabor mais "ruim"
traj_tempo = [tempo_ini]
traj_temp = [temp_ini]

for _ in range(25):
    dtempo = -traj_tempo[-1]
    dtemp = -traj_temp[-1]
    traj_tempo.append(traj_tempo[-1] + 0.1 * dtempo)
    traj_temp.append(traj_temp[-1] + 0.1 * dtemp)

# Plot
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
nota = nota_bolo_z(Tz, Tempz, sabor_ini)
surf = ax.plot_surface(Tz, Tempz, nota, cmap='plasma', alpha=0.8, edgecolor='none')
ax.plot(traj_tempo, traj_temp, nota_bolo_z(np.array(traj_tempo), np.array(traj_temp), sabor_ini),
        color='black', marker='o', markersize=4, label='Caminho da bolinha')
ax.set_title("Gradient Descent no espaço normalizado (z-score)")
ax.set_xlabel("Tempo (z-score)")
ax.set_ylabel("Temperatura (z-score)")
ax.set_zlabel("Nota do Bolo")
ax.legend()
plt.tight_layout()
plt.show()
