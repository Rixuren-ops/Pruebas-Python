import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def proyectar_tiempos(tiempos1, max_contratos, step=1_000_000):
    """
    Realiza una proyección del tiempo de consulta en base al crecimiento observado.
    También calcula el tiempo promedio de consulta por cantidad de contratos.
    :param tiempos1: Lista con los tiempos que tomó cada consulta en milisegundos.
    :param max_contratos: Máxima cantidad de contratos para proyectar.
    :param step: Incremento en la cantidad de contratos (por defecto 1 millón).
    :return: Predicciones de tiempo para los nuevos contratos.
    """
    # Convertir los tiempos a segundos
    tiempos = np.array(tiempos1) 
    contratos = np.arange(1_000_000, (len(tiempos) + 1) * step, step).reshape(-1, 1)
    
    # Generar nuevos contratos solo si max_contratos es mayor que el último valor de contratos
    if max_contratos > contratos[-1]:
        nuevos_contratos = np.arange((len(tiempos) + 1) * step, max_contratos + step, step).reshape(-1, 1)
    else:
        nuevos_contratos = np.array([]).reshape(-1, 1)
    
    # Verificar que los tamaños de los arrays coincidan
    if len(tiempos) != contratos.shape[0]:
        raise ValueError("El tamaño de los tiempos no coincide con el de los contratos.")
    
    # Probar con regresión polinómica para capturar mejor la tendencia
    poly = PolynomialFeatures(degree=2)
    contratos_poly = poly.fit_transform(contratos)
    
    if nuevos_contratos.size > 0:
        nuevos_contratos_poly = poly.transform(nuevos_contratos)
    else:
        nuevos_contratos_poly = np.array([]).reshape(-1, contratos_poly.shape[1])
    
    modelo = LinearRegression()
    modelo.fit(contratos_poly, tiempos)
    
    if nuevos_contratos_poly.size > 0:
        predicciones = modelo.predict(nuevos_contratos_poly)
    else:
        predicciones = []
    
    # Graficar los resultados
    plt.scatter(contratos, tiempos, color='blue', label='Datos reales')
    if len(predicciones) > 0:
        plt.plot(nuevos_contratos, predicciones, color='red', linestyle='--', label='Proyección')
    plt.xlabel('Cantidad de contratos')
    plt.ylabel('Tiempo de consulta (s)')
    plt.legend()
    plt.title('Proyección del tiempo de consulta en base de datos')
    plt.savefig('proyeccion_tiempos.png')  # Guardar la imagen en el directorio actual
    plt.show()
    
    for i in range(10, (max_contratos // step) + 1, 10):
        tiempo_total_acumulado = 0
        for j in range(1, i + 1):
            contratos_multiplo = j * step
            if contratos_multiplo <= max_contratos:
                y_pred = modelo.predict(poly.transform([[contratos_multiplo]]))[0]
                tiempo_total_acumulado += y_pred
        tiempo_total_acumulado_dividido = tiempo_total_acumulado / 4
        print(f"Tiempo total acumulado hasta {i} millones de contratos dividido por 4: {tiempo_total_acumulado_dividido:.2f} segundos")
    

    return predicciones
    
# Datos reales
tiempos1 = [16.789, 20.142, 22.252, 24.981, 29.604, 26.317, 27.494, 29.318, 29.365, 37.310,
            29.408, 36.156, 38.352, 39.881, 41.240, 38.929, 38.633, 43.136, 50.458, 51.802,
            49.862, 42.137, 45.750, 66.222, 49.139, 66.128, 55.697, 71.471]  # Tiempos en segundos

max_contratos = 100_000_000

predicciones = proyectar_tiempos(tiempos1, max_contratos)
print("Proyección de tiempos para nuevos contratos:", predicciones)