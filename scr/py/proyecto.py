# nombre.py
# Hecho por 
# Anthonny Flores Rojas C32975
# Randal Picado Bermudez C36024
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

#Se crea la función matmul_naive que se solicita que recibe las matrices que se van a multiplicar

def matmul_naive(A,B):
    
#Se saca la info de las filas y columnas que se necesitan para hacer la multiplicación
    
    FA = A.shape[0]  #Filas de A
    CA = A.shape[1]  #Columnas de A
    CB = B.shape[1]  #Columnas de B
    
#Se inicializa primero una matriz de puros ceros con las dimensiones

    C = np.zeros((FA, CB))
    
#Se hacen los tres ciclos que nos permiten hacer el producto 
    for i in range (FA):
        for j in range (CB):
            for k in range (CA):
                C[i][j] += A[i][k] * B[k][j]
                
    return C


#Se crea la función matumul_vectorized con funciones de Numpy

def matmul_vectorized(A,B):
    return np.dot(A,B)

#Se nota la diferencia en la longitud y complejidad del código cuando no se usa la herramienta de Numpy 

#Tomemos los casos de ejemplo que salen en la página

A = np.array([[1, 2],[3, 4]])

B = np.array([[5, 6],[7, 8]])

print("Naive:")
print(matmul_naive(A, B))

print("\nVectorized:")
print(matmul_vectorized(A, B))


# Ejercicio 3

#Se implementa la función que se pide que sume, multiplique y divida elemento a elemento de dos matrices A y B

def elementwise_ops(A,B):
    
    variable_epsilon = 1e-8 
    
    return {
        "add" : A + B,
        "mul" : A * B,
        "div" : A / (B+variable_epsilon)
        }

#Se hacen los casos de ejemplo para verificar que todo funcione 

a = np.array([1.0, 2.0])
b = np.array([0.0, 2.0])

result = elementwise_ops(a, b)
print(result)


# Ejercicio 4

#Se implementa la función que se solicita

def reshape_and_transpose(x, B, C, H, W):
    
    #Primero se verifica si el tamaño es correcto
    
    assert len (x) == B * C * H * W
    
    #Se hace el reshape 
    
    reshaped = x.reshape(B,C,H,W)
    
    #Se hace la transpuesta necesari
    
    resultado = reshaped.transpose(0,2,3,1)

    return resultado


#Se usa el ejemplo de prueba para verificar que todo funciona bien

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

B, C, H, W = 1, 2, 3, 4

resultado = reshape_and_transpose(x, B, C, H, W)

print("Shape:", resultado.shape)  # (1, 3, 4, 2)
print(resultado)


# Ejercicio 5
def tensor_reductions(x: np.ndarray, axis: int):
    resultados = {} #Se crea un diccionario para los resultados
    #Se usan las funciones de np para cada operacion, las cuales manejan los axis
    suma = np.sum(x, axis = axis)
    media = np.mean(x, axis = axis)
    maximo = np.max(x, axis = axis)
    pos_max = np.argmax(x, axis = axis)
    #Se agrega cada resultado al diccionario con las respectivas claves
    resultados["sum"] = suma
    resultados["mean"] = media
    resultados["max"] = maximo
    resultados["argmax"] = pos_max
    return resultados

#Caso de prueba que se da en la pagina
x = [[1, 2, 3], 
     [4, 5, 6]]
axis = 1
print(tensor_reductions(x, axis))

# Ejercicio 6

# Ejercicio 7

# Ejercicio 8

#Se implementa la función con los einsum de las operaciones que se piden con dps matrices A y B

def einsum(A,B):
    return {
        "transpose" : np.einsum("ij -> ji", A),
        "sum": np.einsum("ij ->", A), 
        "row_sum" : np.einsum("ij -> i", A),
        "col_sum" : np.einsum("ij -> j", A),
        "matmul":    np.einsum("ik,kj->ij", A, B)
        }

#Hagamos los ejemplos que vienen en la página

A = np.array([[1, 2],[3, 4]])
B = np.array([[5, 6],[7, 8]])

#Impresión estética de resultados

result = einsum(A, B)
for key, val in result.items():
    print(f"{key}:\n{val}\n")


# Ejercicio 9

#Se implementa la función batch que se solicita 

def batch_matmul(Q, K):
    return np.einsum('bhid,bhjd->bhij', Q, K)

#Hacemos el caso de prueba que viene en la página

B, H, S, D = 2, 2, 3, 4
Q = np.random.randn(B, H, S, D)
K = np.random.randn(B, H, S, D)

scores = batch_matmul(Q, K)
print("Shape:", scores.shape)  

#Se verifica que coincide con el bucle manual

bucle_scores = np.zeros((B, H, S, S))
for b in range(B):
    for h in range(H):
        bucle_scores[b, h] = Q[b, h] @ K[b, h].T

print("Resultados iguales:", np.allclose(scores, bucle_scores))



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






