import numpy as np
from typing import Tuple, Optional,Dict

from bitacora1 import one_hot_encode

#Ejercicio 1:

#Pongo la función y los parámetros que se nos pide implementar

def linear_forward(x, w, b):
    
    #Hay que lograr la operación matemática Y = XW + B

    #Para que esto se pueda hacer, se necesita que si o si las dimensiones de las matrices
    #se pueda hacer el producto matricial correspondiente 
    
    #Se sacan las dimensiones necesarias para que el producto matricial tenga sentido 
    
    columna_x = x.shape[1]
    
    fila_w = w.shape[0]

    if (columna_x != fila_w):
        #Si no cumplen con ser iguales pues no se puede hacer el producto 
        print("Las dimensiones no son correctas")
        return None
    else :
        #Si si cumplen, se hace el producto matricial con broadcasting como se pide y se retorna
        multiplicacion_matricial = np.dot(x,w)
        return (multiplicacion_matricial + b)


#Veamos si funciona con el caso de prueba 1 que tira el paper code 

x1 = np.array([[1, 2]])
w1 = np.array([[1, 0],[0, 1]])
b1 = np.array([1, 1])

print(linear_forward(x1,w1,b1))

#Prueba 2

x2 = np.array([[1, 2, 3],[4, 5, 6]])
w2 = np.array([[0.1, 0.2],[0.3, 0.4],[0.5, 0.6]])
b2 = np.array([1.0, 2.0])

print(linear_forward(x2, w2, b2))

#Prueba 3

x3 = np.array([[1, 2]])
w3 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
b3 = np.array([1, 1, 1])

print(linear_forward(x3, w3, b3))


#Ejercicio 2:

def linear_backward(dout, x, w, b):
    
    # Se saca el gradiente respecto a x con la fórmula que nos da el paper code
    dx = np.dot(dout, w.T)
    
    # Se saca el gradiente respecto a w con la fórmula que nos da el paper code
    dw = np.dot(x.T, dout)
    
    # Gradiente respecto a b que es suma de filas de dout con la fórmula de la sumatoria
    db = np.sum(dout, axis=0)
    
    return {"dx": dx, "dw": dw, "db": db}


# Caso de prueba
x    = np.array([[1, 2], [3, 4]])
w    = np.array([[0.5, 0.6], [0.7, 0.8]])
b    = np.array([0.1, 0.2])
dout = np.array([[0.1, 0.2], [0.3, 0.4]])

gradientes = linear_backward(dout, x, w, b)

#Se imprime la prueba 

print("dx:", gradientes["dx"])   
print("dw:", gradientes["dw"])   
print("db:", gradientes["db"])   


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

def sigmoid_ops(x: np.ndarray, dout: np.ndarray):

    x    = np.asarray(x)
    dout = np.asarray(dout)

    # Forward pass, versión numéricamente estable según el signo de x
    # Si x >= 0: σ(x) = 1 / (1 + e^(-x))
    # Si x <  0: σ(x) = e^x / (1 + e^x)  → evita e^(-x) enorme
    out = np.where(x >= 0,
                   1 / (1 + np.exp(-x)),
                   np.exp(x) / (1 + np.exp(x)))

    # Backward pass, derivada σ'(x) = σ(x)(1 - σ(x)), luego regla de la cadena
    dx = dout * out * (1 - out)

    return {"out": out, "dx": dx}

# Caso de prueba
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
dout = np.array([ 0.1,  0.2, 0.3, 0.4, 0.5])

resultado = sigmoid_ops(x, dout)

print("out (forward)")
print( np.round(resultado["out"], 3))

print("dx (backward)")
print( np.round(resultado["dx"], 4))

#Ejercicio 6:

def tanh_ops(x, dout):

    x    = np.asarray(x)
    dout = np.asarray(dout)

    # Forward pass, numpy ya tiene tanh optimizada y numéricamente estable
    out = np.tanh(x)

    # Backward pass, derivada tanh'(x) = 1 - tanh²(x), luego regla de la cadena
    dx = dout * (1 - out ** 2)

    return {"out": out, "dx": dx}


# Caso de prueba

x    = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
dout = np.array([ 0.1,  0.2, 0.3, 0.4, 0.5])

resultado = tanh_ops(x, dout)

print("out (forward)")
print(np.round(resultado["out"], 3))
print("dx (backward)")
print(np.round(resultado["dx"], 4))

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

def sgd_step(w, dw, lr):

    w  = np.asarray(w)
    dw = np.asarray(dw)

    # Actualización SGD: w_nuevo = w - lr * dw
    w_new = w - lr * dw

    return w_new


# Caso de prueba

w  = np.array([[1.0, 2.0], [3.0, 4.0]])
dw = np.array([[0.5, -0.3], [-0.2, 0.1]])
lr = 0.01

w_new = sgd_step(w, dw, lr)

print("w actualizado")
print(np.round(w_new, 3))

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

def xavier_init(shape):

    fan_in, fan_out = shape

    # Límite de la distribución uniforme: sqrt(6 / (fan_in + fan_out))
    limit = np.sqrt(6.0 / (fan_in + fan_out))

    # Se muestrea de U(-limit, limit) con la forma pedida
    w = np.random.uniform(low=-limit, high=limit, size=shape)

    return w


# Caso de prueba

shape = (100, 50)
w = xavier_init(shape)

print("Shape:", w.shape)
print("Min:  ", round(w.min(), 4))
print("Max:  ", round(w.max(), 4))
print("Media:", round(w.mean(), 4))

#Ejercicio 15

def kaiming_init(shape):

    fan_in, fan_out = shape

    # Desviación estándar: sqrt(2 / fan_in)
    std = np.sqrt(2.0 / fan_in)

    # Se muestrea de N(0, std) con la forma pedida
    w = np.random.normal(loc=0.0, scale=std, size=shape)

    return w


# Caso de prueba

shape = (100, 50)
w = kaiming_init(shape)

print("Shape:", w.shape)
print("Media:", round(w.mean(), 4))
print("Std:  ", round(w.std(), 4))

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

#Se nos pide implementar la siguiente función con estos parámetros:

def batchnorm_forward(x, gamma, beta, eps=1e-5, momentum=0.9,running_mean=None, running_var=None, train=True):

    #Se hace todo un vector de numpy 
    
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    N, D = x.shape

    # Se crean las matrices de ceros para el promedio y la varianza
    
    if running_mean is None:
        running_mean = np.zeros(D)
    if running_var is None:
        running_var = np.zeros(D)

    #Se hace el paso del entrenamiento 

    if train:
        #Se saca el promedio y la varianza
        
        mean = np.mean(x, axis=0)                        
        var  = np.mean((x - mean) ** 2, axis=0)  
        
        #Se normaliza   
        
        x_norm = (x - mean) / np.sqrt(var + eps)   
        
        #Se hace la escala y el shift? No sé la traducción XD
        
        out    = gamma * x_norm + beta                    


        #Se actualizan los runnign mean y el running var
        
        new_running_mean = momentum * running_mean + (1 - momentum) * mean
        new_running_var  = momentum * running_var  + (1 - momentum) * var

    else:
        # en lugar de calcular la media y varianza del batch actual, usa las running_mean 
        # y running_var que se fueron acumulando durante el train
        
        var    = running_var
        x_norm = (x - mean) / np.sqrt(var + eps)
        out    = gamma * x_norm + beta

        new_running_mean = running_mean
        new_running_var  = running_var

    cache = {
        'x':      x,
        'x_norm': x_norm,
        'mean':   mean,
        'var':    var,
        'gamma':  gamma,
        'eps':    eps,
    }

    return out, cache, new_running_mean, new_running_var

#Caso de prueba 

x            = np.array([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0]])   
gamma        = np.array([1.0, 1.0])     # Shape (2,)
beta         = np.array([0.0, 0.0])     # Shape (2,)
eps          = 1e-5
momentum     = 0.9
running_mean = None
running_var  = None

#Llamamos la función 

out, cache, new_running_mean, new_running_var = batchnorm_forward(
    x, gamma, beta,
    eps=eps,
    momentum=momentum,
    running_mean=running_mean,
    running_var=running_var,
    train=True
)

#Imprimimos los valores obtenidos y los esperados

print("=== Paso 1: Mean ===")
print("Obtenido :", cache['mean'])
print("Esperado : [3.0, 4.0]\n")

#Varianza
print("=== Paso 2: Var ===")
print("Obtenido :", cache['var'])
print("Esperado : [2.6667, 2.6667]\n")

#Normalización
print("=== Paso 3: x_norm ===")
print("Obtenido :\n", cache['x_norm'])
print("Esperado :\n [[-1.225, -1.225], [0., 0.], [1.225, 1.225]]\n")

#escalada
print("=== Paso 4: out ===")
print("Obtenido :\n", out)
print("Esperado :\n [[-1.225, -1.225], [0., 0.], [1.225, 1.225]]\n")

#Running stats
print("=== Paso 5: Running stats ===")
print("new_running_mean obtenido :", new_running_mean)
print("new_running_mean esperado : [0.3, 0.4]")
print("new_running_var  obtenido :", new_running_var)
print("new_running_var  esperado : [0.2667, 0.2667]\n")


#Ejercicio 19:

#Se hace la función con los parámetros que se piden, el cache es un diccionario 

def batchnorm_backward(dout, cache):

    # Extraer valores guardados en el diccionario cache 
    x      = cache['x']       
    x_norm = cache['x_norm']  
    mean   = cache['mean']    
    var    = cache['var']     
    gamma  = cache['gamma']   
    eps    = cache['eps']       

    #Se obtiene el dbeta primero
    
    dbeta = np.sum(dout, axis=0)            

    #Luego el dgamma

    dgamma = np.sum(dout * x_norm, axis=0)  

    # Gradiente que llega a x_norm antes de escalar con gamma
    
    dhat_x = dout * gamma                         

    # Desviación estándar usada en el forward
    
    std = np.sqrt(var + eps)                       

    # Se hace la corrección con la media
    
    dmu = np.mean(dhat_x, axis=0)                 

    # Se hace la corrección con la varianza
    
    dvar = np.mean(dhat_x * (x - mean), axis=0)   # (D,)

    # Gradiente final combinando las tres correcciones hechas anteriormente 
    
    dx = (1 / std) * (dhat_x - dmu - x_norm * dvar)   

    return {
        'dx':     dx,       
        'dgamma': dgamma,   
        'dbeta':  dbeta,    
    }


# Caso de prueba                                                   

cache = {
    'x':      np.array([[1.0, 2.0], [3.0, 4.0],[5.0, 6.0]]),    
    'x_norm': np.array([[-1.225, -1.225], [ 0.0,   0.0  ],[ 1.225, 1.225]]),
    'mean':   np.array([3.0, 4.0]),       
    'var':    np.array([2.67, 2.67]),     
    'gamma':  np.array([1.0, 1.0]),       
    'eps':    1e-5                        
}

dout = np.array([[0.1, 0.2],[0.3, 0.4],[0.5, 0.6]])

#Se crean los gradientes con la función implementada y se imprimen bonitos con lo obtenido y lo esperado

gradientes = batchnorm_backward(dout, cache)

print("=== dbeta ===")
print("Obtenido:", gradientes['dbeta'])
print("Esperado: [0.9, 1.2]\n")
print("=== dgamma ===")
print("Obtenido:", gradientes['dgamma'])
print("Esperado: [0.49, 0.49]\n")
print("=== dx ===")
print("Obtenido:\n", gradientes['dx'])
print("Esperado: valores muy pequeños\n")


#Ejercicio 20:

#Se define la función que se solicita 

def layernorm_forward(x, gamma, beta, eps=1e-5):

# Se tira todo a numpy como en el 18 
    x     = np.array(x,     dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta  = np.array(beta,  dtype=float)

# Promedio y varianza POR MUESTRA (axis=1) en lugar de por batch (axis=0)
# keepdims=True mantiene la forma (N,1) para que el broadcasting funcione

    mean = np.mean(x,             axis=1, keepdims=True)  
    var  = np.mean((x - mean)**2, axis=1, keepdims=True)  

    # Normalización: igual que ej. 18, pero mean y var son (N,1) no (D,)

    x_norm = (x - mean) / np.sqrt(var + eps)  # (N,D)

    # Lo escalamos con en el ejercicio 18 
    
    out = gamma * x_norm + beta  # (N,D)

    # Cache con los mismos campos que el ej. 18 y 19
    cache = {
        'x':      x,
        'x_norm': x_norm,
        'mean':   mean,
        'var':    var,
        'gamma':  gamma,
        'eps':    eps,
    }

    return out, cache

#Se define la función que se solicita del ejercicio 19 

def layernorm_backward(dout, cache):

    # Se extraen los valores del cache igual que en el ejercicio 19
    
    x      = cache['x']
    x_norm = cache['x_norm']
    mean   = cache['mean']
    var    = cache['var']
    gamma  = cache['gamma']
    eps    = cache['eps']

    std = np.sqrt(var + eps)  

    # dbeta y dgamma: igual que ej. 19, se suman sobre el batch (axis=0)
    
    dbeta  = np.sum(dout,          axis=0)  # (D,)
    dgamma = np.sum(dout * x_norm, axis=0)  # (D,)

    # Gradiente que llega a x_norm: igual que ejercicio 19
    
    dhat_x = dout * gamma  

    # Correcciones por media y varianza, pero sobre axis=1
    # porque en LayerNorm la media y varianza se calcularon sobre las features
    
    dmu  = np.mean(dhat_x,               axis=1, keepdims=True)  
    dvar = np.mean(dhat_x * (x - mean),  axis=1, keepdims=True)  

    # Gradiente final, mismo toque que en el ejercicio 19
    
    dx = (1 / std) * (dhat_x - dmu - x_norm * dvar)  

    return {
        'dx':     dx,      
        'dgamma': dgamma,  
        'dbeta':  dbeta,   
    }


# Caso de prueba

x     = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
gamma = np.array([1.0, 1.0, 1.0])
beta  = np.array([0.0, 0.0, 0.0])

out, cache = layernorm_forward(x, gamma, beta)

#Impresión de todo lo que lleva esa función forward, se pone lo obtenido y lo esperado para comparar

print("===Forward===")
print("mean   obtenida:", cache['mean'].ravel(), "esperado: [2.0, 5.0]")
print("var    obtenida:", cache['var'].ravel(),  "esperado: [0.6667, 0.6667]")
print("x_norm obtenida:\n", cache['x_norm'])
print("out    obtenida:\n", out)

#Impresión de todo lo que lleva esa función backeward, se pone lo obtenido y lo esperado para comparar

dout  = np.ones((2, 3))
gradientes = layernorm_backward(dout, cache)

print("\n=== Backward ===")
print("dbeta  obtenida:", gradientes['dbeta'],  "esperado: [2. 2. 2.]")
print("dgamma obtenida:", gradientes['dgamma'], "esperado: [-2.4495  0.      2.4495]")
print("dx     obtenida:\n", gradientes['dx'],  "esperado: valores tirando a cero")





