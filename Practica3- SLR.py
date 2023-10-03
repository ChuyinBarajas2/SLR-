import numpy as np
import matplotlib.pyplot as plt



listaX=[1.2, 1.4, 1.6, 2.1, 2.3, 3, 3.1, 3.3, 3.3, 3.8, 4, 4.1, 4.1, 4.2, 4.6, 5, 5.2, 5.4, 6, 6.1, 6.9, 7.2, 8, 8.3, 8.8, 9.1, 9.6, 9.7, 10.4, 10.6]
listaY=[39344, 46206, 37732, 43526, 39892, 56643, 60151, 54446, 64446, 57190, 63219, 55795, 56958, 57082, 61112, 67939, 66030, 83089, 81364, 93941, 91739, 98274, 101303, 113813, 109432, 105583, 116970, 112636, 122392, 121873]

x = np.array(listaX)
y = np.array(listaY)


x1 = np.random.uniform(-1, 1)
x2 = np.random.uniform(-1, 1)

rate = 0.01
num_iterations = 100
f_values = []  


# Función a optimizar
def funcion_a_optimizar(x1, x2):
    return 10 - np.exp(-(x1**2 + 3 * x2**2))

# Descenso del gradiente para la función a optimizar
for i in range(num_iterations):
    gradiente_x1 = 2 * x1 * np.exp(-(x1**2 + 3 * x2**2))
    gradiente_x2 = 6 * x2 * np.exp(-(x1**2 + 3 * x2**2))
    
    x1 -= rate * gradiente_x1
    x2 -= rate * gradiente_x2

    iteracion_f = funcion_a_optimizar(x1, x2)
    f_values.append(iteracion_f)  # Almacenar los valores de f(x1, x2)


#Regresion lineal 
b = 0.0
w = 0.0
mse_valores = []

def funcion_pre(x,b,w):
    return b+w*x

for i in range(num_iterations):
    prediccion = funcion_pre(x,b,w)

    mse = ((prediccion-listaY)**2).mean()
    mse_valores.append(mse)

    gradiente_b = 2 * np.mean(y - prediccion)
    gradiente_w = 2 * np.mean((y - prediccion) * x)

    b -= rate * gradiente_b
    w -= rate * gradiente_w


#Desenso del gradiante
print("Valor mínimo:", iteracion_f)
print("Posición del mínimo (x1, x2):", (x1, x2))

#Regresion lineal en base al desenso del gradiante
plt.plot(range(num_iterations), f_values)
plt.xlabel('Iteración')
plt.ylabel('f(x1, x2)')
plt.title('Evolución de la función durante el descenso del gradiente')
plt.show()

#Mse
print("Valor mínimo del MSE:", mse)
print("Parámetros aprendidos:")
print("Término de sesgo (b):", b)
print("Pendiente (w):", w)