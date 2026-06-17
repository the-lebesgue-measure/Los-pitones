import numpy as np
from typing import Tuple, Optional,Dict

from bitacora1 import one_hot_encode

#Ejercicio 1:


#Ejercicio 2:


#Ejercicio 3:

def relu_forward(x: np.ndarray) -> np.ndarray:
    """
    Computes ReLU(x) = max(0, x).
    """

    modified_x = np.asarray(x)#Se asegura que el parámetro esté en el tipo de dato correcto.
    desired_coordinates = np.where(modified_x<0) #Se ubican las coordenadas de todos los puntos menores que cero.
    modified_x[desired_coordinates]=0 #Se reemplazan todos los números ubicados en las coordenadas encontradas en la línea anterior por cero.
    pass
    return modified_x


#Ejercicio 4:

def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Computes dx for ReLU.
    """
    
    modified_dout = np.asarray(dout)
    modified_x = np.asarray(x)#Se asegura que los datos sean del tipo correcto.

    normalized_x = modified_x>0 #Se crea un tensor que reemplaza cada valor positivo por un True (equivalente a 1 en Numpy) y los valores no positivos por False (equivalente a 0 en Numpy).


    dx=modified_dout*normalized_x#Se realiza el producto entrada por entrada de acuerdo a la explicación brindada por el enunciado.

    pass

    return dx #Se devuelve el tensor deseado.

#Ejercicio 5:

#Ejercicio 6:

#Ejercicio 7:

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float | np.ndarray]:
    """
    Computes MSE loss and gradient.
    """
    
    y_pred = np.asarray(y_pred) # Se rectrifica el tipo correcto np
    y_true = np.asarray(y_true) 

    errors = y_pred - y_true # Error (diferencia)

    squared_errors = errors ** 2 # Error cuadratico

    loss = np.mean(squared_errors) # Media del error cuadratico

    N, D = y_true.shape # Dimensiones de los vectores 


    
    dx = (2 / (N * D)) * errors # Gradiente del MSE

    answer = { # Diccionario de las respuestas

        'loss' : loss,
        'dx' : dx,
    }

    return answer   



#Ejercicio 8:

def bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float | np.ndarray]:
    """
    Computes BCE loss and gradient.
    """

    error = 1e-9 #Épsilon que se usará durante los cálculos con logaritmos con el fin de evitar desbordamientos.

    y_pred = np.asarray(y_pred)  #Se asegura que el tensor con probabilidades obtenidas mediante el modelo sea del tipo correcto.
    y_true = np.asarray(y_true)    #Se asegura que el tensor con probabilidades empíricas sea del tipo correcto.

    complement_y_pred = 1-y_pred #Vector con el complemento de las probabilidades almacenadas en y_pred
    complement_y_true = y_true<=0#Análogo a lo anterior para y_true.

    loss_vector = y_true*np.log(y_pred+error) + complement_y_true*np.log(complement_y_pred+error) #Se usa la fórmula dada por el enunciado para calcular el vector que registra las pérdidas de manera vectorizada.

    loss_mean = np.abs(np.mean(loss_vector)) #Se calcula la media del valor absoluto de las pérdidas.

    dx = (y_pred - y_true)/len(y_true) #Se almacena en un tensor el conjunto de diferencias entre las probabilidades obtenidas del modelo y las probabilidades verdaderas. A su vez, se divide entre la cantidad de probabilidades que había que predecir (esto pues así lo exige la fórmula vista en el enunciado).

    answer = {

        'loss' : loss_mean,
        'dx' : dx,
    
    }#Se crea el diccionario que servirá como el output de la función.

    pass
    return answer

#Ejercicio 9:

def categorical_ce_backward(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Computes gradient of Softmax+CE w.r.t logits.
    """
    
    probs = np.asarray(probs) # Se rectrifica el tipo correcto np
    targets = np.asarray(targets) 
    num_classes = probs.shape[1] # Se toma el num_classes (dimenciones a lo ancho) para one hot

    one_hot = one_hot_encode(targets, num_classes) # Se utiliza el one_hot ya existente 

    gradient = probs - one_hot # Se calcula la diferencia entre probs y one hot

    N = probs.shape[0] # Se toma el N

    dx = gradient/N # Se dividide por el N

    return dx



#Ejercicio 10:

#Ejercicio 11:

def momentum_step(w: np.ndarray, dw: np.ndarray, v: np.ndarray, lr: float, momentum: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs SGD with momentum update.
    Returns (w_new, v_new).
    """

    if v is None:
        v = np.zeros_like(w) # Se inicializa como 0s si no hay v

    v_new = momentum * v + dw # V nuevo

    w_new = w - lr * v_new # W nuevo 

    return w_new, v_new


#Ejercicio 12:

def rmsprop_step(w: np.ndarray, dw: np.ndarray, cache: np.ndarray, lr: float, decay: float = 0.99, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    RMSProp update. Returns (w_new, cache_new).
    """

    if cache is None:
        cache = np.zeros_like(w) # Se inicializa como 0s si no hay s
    
    dw_squared = dw ** 2 # Valores dw cuadrados

    cache_new = decay * cache + (1 - decay) * dw_squared # Se calcula el s nuevo

    adaptive_lr = lr / (np.sqrt(cache_new) + eps) # Se calcula el eta / raiz de s nuev0 + epsilon por el gradiente de L

    w_new = w - adaptive_lr * dw # Calculo del w nuevo

    return w_new, cache_new


#Ejercicio 13:

def adam_step(w: np.ndarray, dw: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adam update. Returns (w_new, m_new, v_new).
    t is the current timestep (1-indexed).
    """
    if m is None:
        m = np.zeros_like(dw) # Se inicializa como 0s si no hay momento primero

    if v is None:
        v = np.zeros_like(dw) # Se inicializa como 0s si no hay momento segundo

    m_new = beta1 * m + (1 - beta1) * dw # Se calcula el nuevo m
    v_new = beta2 * v + (1 - beta2) * (dw ** 2) # Se calcula el nuevo v


    beta1_power = beta1 ** (t + 1) # Se calcula el beta para m sombrero
    beta2_power = beta2 ** (t + 1) # Se calcula el beta para v sombrero

    m_hat = m_new / (1 - beta1_power) # Se calcula m sombrero
    v_hat = v_new / (1 - beta2_power) # Se calcula v sombrero

    adaptive_lr = lr / (np.sqrt(v_hat) + eps) # Calculo del Lr / rais de theta + epsilon

    w_new = w - adaptive_lr * m_hat # Se calucla el nuevo w
    
    return w_new, m_new, v_new


#Ejercicio 14:

#Ejercicio 15:

#Ejercicio 16:

def dropout_forward(x: np.ndarray, p: float, train: bool = True, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverted dropout forward. Returns (out, mask).
    """
    
    if not train:
        return (x,None)#El caso en el que no se desea entrenar al modelo.
    
    x = np.asarray(x) #Se asegura que el parámetro sea del tipo correcto.

    shape = np.shape(x) #Se almacenan las dimensiones del tensor recibido en el parámetro

    drop_probability = p #Cambio de nombre a la probabilidad p para hacer el código más legible.

    mask = np.random.choice(a=(True,False),size=shape,p=[1-drop_probability,drop_probability]) #Se crea un tensor con las mismas dimensiones del parámetro 'x' de tal manera que cada coordenada tiene una probabilidad 'p' de contener un False y '1-p' de contener True.

    masked = mask*x #Se hace la multiplicación entrada por entrada de ambos tensores.
    
    if drop_probability < 1:
        out = masked / (1 - drop_probability) #Si drop_probability < 1, entonces el cociente 1/(1-drop_probability) está bien definido, por lo que se puede hacer la multiplicación escalar entre dicho valor y el tensor en cuestión.
    else:
        out = masked #Caso de frontera: Si drop_probability = 1 significa que todas las entradas de la matriz fueron reemplazadas por cero. En ese caso, no hace falta multiplicar a la matriz 'masked' por ningún escalar. Además, el cociente 1/(1-drop_probability) no está definido cuando drop_probability = 1.

    pass
    return(out,mask)#Se devuelve la tupla deseada.

#Ejercicio 17:

def dropout_backward(dout: np.ndarray, mask: Optional[np.ndarray], p: float, train: bool = True) -> np.ndarray:
    """
    Backward pass for inverted dropout.
    """

    dout = np.asarray(dout)#Se asegura que el parámetro sea del tipo de dado correcto.
    if not train:
        return dout#El caso en el que no se desea entrenar al modelo.
    mask = np.asarray(mask)#Se asegura que el parámetro sea del tipo de dado correcto.
    drop_probability = p #Cambio de nombre a la probabilidad p para hacer el código más legible.
    masked_grad = mask * dout #Se hace la multiplicación entrada por entrada de ambos tensores de acuerdo a la fórmula expuesta en el enunciado.

    if drop_probability < 1:
        dx = masked_grad / (1 - drop_probability)#Si drop_probability < 1, entonces el cociente 1/(1-drop_probability) está bien definido, por lo que se puede hacer la multiplicación escalar entre dicho valor y el tensor en cuestión.
    else:
        dx = masked_grad #Caso de frontera: Si drop_probability = 1 significa que todas las entradas de la matriz fueron reemplazadas por cero. En ese caso, no hace falta multiplicar a la matriz 'masked' por ningún escalar. Además, el cociente 1/(1-drop_probability) no está definido cuando drop_probability = 1.
    pass
    return dx #Se retorna el tensor deseado.

#Ejercicio 18:

#Ejercicio 19:

#Ejercicio 20:


