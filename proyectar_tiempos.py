import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def proyectar_tiempos(tiempos, max_contratos, step=1_000_000):
    """
    Realiza una proyección del tiempo de consulta en base al crecimiento observado.
    También calcula el tiempo promedio de consulta por cantidad de contratos.
    :param tiempos: Lista con los tiempos que tomó cada consulta.
    :param max_contratos: Máxima cantidad de contratos para proyectar.
    :param step: Incremento en la cantidad de contratos (por defecto 1 millón).
    :return: Predicciones de tiempo para los nuevos contratos.
    """
    contratos = np.arange(1_000_000, (len(tiempos) + 1) * step, step).reshape(-1, 1)
    nuevos_contratos = np.arange((len(tiempos) + 1) * step, max_contratos + step, step).reshape(-1, 1)
    
    tiempos = np.array(tiempos) / 1000  # Convertir a miles de segundos para la gráfica
    
    # Probar con regresión polinómica para capturar mejor la tendencia
    poly = PolynomialFeatures(degree=2)
    contratos_poly = poly.fit_transform(contratos)
    nuevos_contratos_poly = poly.transform(nuevos_contratos)
    
    modelo = LinearRegression()
    modelo.fit(contratos_poly, tiempos)
    predicciones = modelo.predict(nuevos_contratos_poly)
    
    # Graficar los resultados
    plt.scatter(contratos, tiempos, color='blue', label='Datos reales')
    plt.plot(nuevos_contratos, predicciones, color='red', linestyle='--', label='Proyección')
    plt.xlabel('Cantidad de contratos')
    plt.ylabel('Tiempo de consulta (miles de s)')
    plt.legend()
    plt.title('Proyección del tiempo de consulta en base de datos')
    plt.show()
    
    # Calcular el tiempo promedio de consulta por cantidad de contratos
    total_tiempo = np.sum(tiempos * 1000)  # Convertir de nuevo a segundos para el cálculo
    tiempo_promedio = total_tiempo / (len(tiempos) * step / 1_000_000)
    tiempo_promedio_min = tiempo_promedio / 60  # Convertir a minutos
    
    print(f"Tiempo total para {len(tiempos) * step / 1_000_000} millones de contratos: {total_tiempo:.2f} segundos")
    print(f"Tiempo promedio por millón de contratos: {tiempo_promedio:.2f} segundos ({tiempo_promedio_min:.2f} minutos)")
    
    # Mostrar proyecciones para cantidades específicas de contratos
    for cantidad in range(10_000_000, max_contratos + 1, 10_000_000):
        tiempo_proyectado = modelo.predict(poly.transform([[cantidad]]))[0] * 1000  # Convertir a segundos
        print(f"Proyección de tiempo para {cantidad / 1_000_000:.0f} millones de contratos: {tiempo_proyectado:.2f} segundos ({tiempo_proyectado / 60:.2f} minutos)")
    
    return predicciones

# Datos reales
tiempos = [16789, 20142, 22252, 24981, 29604, 26317, 27494, 29318, 29365, 37310,
           29408, 36156, 38352, 39881, 41240, 38929, 38633, 43136, 50458, 51802,
           49862, 42137, 45750, 66222, 49139, 66128, 55697, 71471]  # Tiempos en segundos

max_contratos = 40_000_000

predicciones = proyectar_tiempos(tiempos, max_contratos)
print("Proyección de tiempos para nuevos contratos:", predicciones)
