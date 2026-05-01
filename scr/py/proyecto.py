# nombre.py
# Hecho por 
# Anthonny Flores Rojas C32975
# ...
# ...
# ...

# Librerias
import numpy as np
from typing import Tuple,Dict

# Ejercicio 1
def broadcast_ops(X: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:

    broad: np.ndarray = [] # Se inicializa la lista de np
 
    sum = X + b # Suma b en cada columna

    w = w[:, np.newaxis] # Cambia el shape de (2,) a (2,1)

    broad = sum * w # Realiza el producto punto

    return broad

# Ejercicio 2

# Ejercicio 3

# Ejercicio 4

# Ejercicio 5

# Ejercicio 6

# Ejercicio 7

# Ejercicio 8

# Ejercicio 9

# Ejercicio 10
def grad_sum(grad_y: float, x_shape: Tuple[int]) -> np.ndarray:
    
    grad_x: np.ndarray = np.full(x_shape, grad_y)  # Se toma la matriz de dimenciones x_shape con valores grad_y

    return grad_x

# Ejercicio 11
def grad_matmul(grad_C: np.ndarray, A: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:

    grad_A: np.ndarray = grad_C @ B.T # Se toma el gradiente por definicion

    grad_B: np.ndarray = A.T @ grad_C

    dict: Dict = { # Se crea el diccionario
        "grad_A": grad_A,
        "grad_B": grad_B
    }

    return dict

# Ejercicio 12
def one_hot_encode(indices: np.ndarray, num_classes: int) -> np.ndarray:
    
    one_hot_vector: np.ndarray = np.eye(num_classes)[indices] # Por cada indice de la indentidad se llama ese indice a un nuevo vector

    return one_hot_vector

# Ejercicio 13
def softmax(x: np.ndarray) -> np.ndarray:

    x = x - np.max(x, axis = 1, keepdims = True) # Se resta el maximo para cada vector
    
    x = np.exp(x) # Se exponencian

    x = x/np.sum(x, axis = 1, keepdims = True) # Se normalizan

    return x

# Ejercicio 14

# Ejercicio 15






