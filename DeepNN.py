import numpy as np
import csv
import click

def leeDatos(nombre):
    X = []
    Y = []
    with open(nombre, newline = '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = ',')
        for fila in spamreader:
            tamano = len(fila) - 1
            X.append((fila[:tamano]))
            Y.append(fila[tamano:][0])
    Y = np.asarray(Y,dtype=np.int32)
    X = np.asmatrix(X,dtype=np.float32)
    return X.T, Y

def mapea_clases(arreglo):
    narreglo = []
    nclases = np.unique(arreglo)
    nclases = nclases.tolist()
    tam = len(nclases)    
    for item in arreglo:
        for clas in nclases:
            if clas == item:
                 pos = nclases.index(clas)
                 trans = [0] * tam
                 trans.pop(pos)
                 trans.insert(pos,1)
                 narreglo.append(trans)
    return np.asmatrix(narreglo)


def softmax(z):
    expA = np.exp(z)
    t = expA.sum(axis=0)
    return expA/t

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_deri(A):
    return np.multiply(A,(1 - A))


def inicializa_parametros(layer_dims):
    
    np.random.seed(3)
    parametros = {}
    L = len(layer_dims)

    for l in range(1, L):
        parametros['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parametros['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parametros['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parametros['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parametros

def linear_forward(A, W, b,l):
    linear = {}
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    linear["Z" + str(l)] = Z
    linear["W" + str(l)] = W
    linear["b" + str(l)] = b    
    return Z, linear


def linear_activation_forward(A_prev, W, b,l, activation):
    activacion = {}
    if activation == "sigmoid":
        Z, linear2 = linear_forward(A_prev, W, b,l)
        A = sigmoid(Z)
    
    elif activation == "softmax":
        Z, linear2 = linear_forward(A_prev, W, b,l)
        A = softmax(Z)
    
    activacion.update(linear2)
    activacion["A" + str(l)] = A
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    return A, activacion


def L_model_forward(X, parametros):

    cach = {}
    cach["A0"] = X
    A = X
    L = len(parametros) // 2
    l = 0
    for l in range(1, L):
        A_prev = A 
        A, datos = linear_activation_forward(A_prev, 
                                             parametros['W' + str(l)], 
                                             parametros['b' + str(l)],
                                             l, 
                                             activation='sigmoid')
        cach.update(datos)
    
    AL, datos = linear_activation_forward(A, 
                                          parametros['W' + str(L)], 
                                          parametros['b' + str(L)],
                                          l+1, 
                                          activation='softmax')
    
    cach.update(datos)          
    return AL, cach

def compute_cost(AL, Y):

    m = Y.shape[1]    
    cost = np.sum(np.multiply(Y, np.log(AL)),axis=0)
    cost = (-1 / m) * np.sum(cost)
    
    cost = np.squeeze(cost)
    
    return cost

def backward_propagation(cache, X, Y):

    m = X.shape[1]

    W1 = cache["W1"]
    W2 = cache["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2= A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), np.multiply(A1,(1 - A1)))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def dZN(W, dZPost, A):
    dA = np.dot(W.T, dZPost)
    return np.multiply(dA,sigmoid_deri(A))

def dWbN(dZ, A, m):
    W = (1 / m) * np.dot(dZ, A.T)
    b = (1 / m) * np.sum(dZ, axis=1)
    return W, b

def backpropagation_NL(cache, X, Y):

    grads = {}
    m = X.shape[1]

    L = len(cache.items()) // 4
    dZ = cache["A" + str(L)] - Y #derivada parcial crossentropy respecto a softmax
    for l in reversed(range(1, L + 1)):
        Aprev = cache["A" + str(l - 1)]
        Wpost = cache["W" + str(l)]

        dW, db = dWbN(dZ, Aprev, m)
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
        dZ = dZN(Wpost, dZ, Aprev)
    return grads



def update_parameters(cache, grads, learning_rate=.02):
    parametros = {}
    L = len(cache.items()) // 4

    for l in range(1,L + 1):
        WN = cache["W" + str(l)]
        bN = cache["b" + str(l)]
        dWN = grads["dW" + str(l)]
        dbN = grads["db" + str(l)]

        WN = WN - learning_rate * dWN
        bN = bN - learning_rate * dbN
        
        parametros["W" + str(l)] = WN
        parametros["b" + str(l)] = bN
    
    return parametros

def progress_bar(costo, epocas):
    print('{0}>> {1}'.format('='*(costo//(epocas//100)),costo),end="\r")


@click.command()
@click.option('-c','--capas', default='a', type= str, help='Numero de capas ocultas ej. --capas= 245')
@click.option('-lr','--learning', default=.02, help='Learning rate de la red neuronal')
@click.option('-ep','--epocas', default=500, help='numero de epocas en la red neuronal')
@click.option('-dt','--data',required=True, type=str,  help='Nombre del conjunto de entrenamiento')
@click.option('--prueba', help='Nombre del conjunto de pruebas')
def main(capas, learning, data, prueba, epocas):
    X, Y = leeDatos(data)
    Y = mapea_clases(Y)
    Y = Y.T
    capa = []
    capa.append(X.shape[0])
    if capas == 'a':
        capa.append(X.shape[0])
    else:
        for i in capas:
            capa.append(int(i))
    capa.append(Y.shape[0])
    parametros = inicializa_parametros([X.shape[0],X.shape[0],Y.shape[0]])
    for i in range(0, epocas +1):
        AL, caches = L_model_forward(X, parametros)
        cost = compute_cost(AL, Y)
        progress_bar(i,epocas)
        grads = backpropagation_NL(caches, X, Y)
        parametros = update_parameters(caches, grads)
    print("\n")
    X, Y = leeDatos("prueba.csv")
    AL, caches = L_model_forward(X, parametros )
    print(np.where(AL == np.amax(AL, axis=0)))


if __name__ == "__main__":
    main()
    """
    X, Y = leeDatos("datos.csv")
    Y = mapea_clases(Y)
    Y = Y.T
    parametros = inicializa_parametros([X.shape[0],X.shape[0],Y.shape[0]])
    for i in range(0, 10000):
        AL, caches = L_model_forward(X, parametros)
        cost = compute_cost(AL, Y)
        print(cost)
        grads = backpropagation_NL(caches, X, Y)
        parametros = update_parameters(caches, grads)

    X, Y = leeDatos("prueba.csv")
    AL, caches = L_model_forward(X, parametros )
    print(np.where(AL == np.amax(AL, axis=0)))
    """