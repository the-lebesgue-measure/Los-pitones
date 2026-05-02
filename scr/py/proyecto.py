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
def compute_norms(x: np.ndarray) -> Dict[str, np.ndarray]:
    resultados = {} #Diccionario para resultados
    l1 = np.sum(np.abs(x), axis = 1) #se aplica el valor absoluto a cada elemento del tensor y luego se suman los valores a lo largo del 
    #axis=1, lo que permite obtener una norma por cada vector.
    l2 = np.sqrt(np.sum(np.square(x), axis=1)) #se elevan los elementos al cuadrado, se suman por filas y se aplica la raíz cuadrada.
    #Se agregan resultados al diccionario
    resultados["l1"] = l1
    resultados["l2"] = l2
    return resultados

#Se prueba con el ejemplo de la pagina

x2 = [[3, 4], 
     [1, -1]]
print(compute_norms(x2))


# Ejercicio 7
def vector_products(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    #se convierten las entradas a arreglos de NumPy
    a = np.asarray(a)
    b = np.asarray(b)
    #Se genera diccionario de resultados
    resultados = {}
    #Para el producto punto, se realiza una multiplicación elemento a elemento entre los vectores y luego se suman los resultados a lo largo del axis=1
    dot = np.sum(a*b, axis=1)
    #Para el producto cruz, se utiliza la función np.cross, que calcula directamente el vector perpendicular para cada par. 
    cross = np.cross(a, b)
    #Finalmente, los resultados se almacenan en un diccionario con las claves "dot" y "cross".
    resultados["dot"] = dot
    resultados["cross"] = cross
    return resultados

#Caso de prueba
a = [[1, 0, 0]]
b = [[0, 1, 0]]

print(vector_products(a, b))

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

def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes mean cross-entropy loss.
    
    Args:
        probs: (N, C) probabilities
        targets: (N,) integer class indices
        
    Returns:
        Scalar float loss
    """
    #probs = np.asmatrix(probs)

    targeted_probabilities = np.take_along_axis(probs,np.transpose(np.asmatrix(targets)),axis=1) 
    #La línea anterior toma la transposición de "targets" y se aplica la función "take_along_axis" a lo largo de las filas (axis = 1) 
    #de la matriz "probs" con el fin de extraer los elementos de la i-ésima fila asociados al número que se encuentra en la i-ésima 
    #posición del vector "targets"  

    negative_log_probabilities = -np.log(targeted_probabilities+1e-9)
    #Se aplica la función logaritmo natural entrada por entrada y a su vez se toma su negativo. El factor +1e-9 era un requisito del 
    #ejercicio.

    loss = np.mean(negative_log_probabilities)
    #Se toma la mea de los valores de los logaritmos negativos obtenidos anteriormente y se calcula la media, lo que por definición 
    #devuelve la pérdida (loss) que se deseaba encontrar.

    pass

    return loss

# Ejercicio 15

def log_sum_exp(x: np.ndarray) -> np.ndarray:
    """
    Computes log(sum(exp(x))) stably along the last axis.
    
    Args:
        x: Input (N, D)
        
    Returns:
        Result (N,)
    """
    maximum_of_rows =np.apply_along_axis(np.max,1,x) 
    
    #Se construye un vector tal que la i-ésima entrada contiene al máximo de la i-ésima fila de la matriz original.
    

    shifted_matrix=np.transpose(np.transpose(x)-maximum_of_rows) 
    #Estabiliza. Es necesario hacer la transposicion una vez, pues la resta por defecto agarra la columna i de la matriz y 
    #le resta la entrada i del vector

    exponentiaded_matrix= np.exp(shifted_matrix) 
    #Se le aplica la función exp(x) entrada por entrada a la matriz obtenida en la línea anterior.

    sum_of_rows = np.sum(exponentiaded_matrix,axis=1)
    #Se construye un vector donde la i-ésima entrada contiene a la suma de los elementos de la i-esima fila de la matriz obtenida en la línea 
    #anterior

    log_sum_exp = maximum_of_rows + np.log(sum_of_rows)
    #Se le aplica la función logaritmo natural entrada por entrada al vector anterior y se le suma los máximos de cada fila, obtenidos en la 
    #primera linea. Dicha suma obtiene lo deseado pues, por construcción, ambos vectores están definidos siguiendo el orden
    #de las filas de la matriz original
    
    pass
    
    return log_sum_exp






