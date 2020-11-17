import numpy as np
import csv
import click
import Orange as og
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def validacionCruzada(X,Y,k,epocas, learning):
    x = X.T
    y = Y.T
    precision = []
    kf = KFold(n_splits=k, shuffle=True, random_state=7)
    i = 1
    for train_f,test_f in kf.split(x,y):
        entrenamiento = np.take(x,train_f,axis=0)
        y_train = np.take(y,train_f)
        y_train = y_train.tolist()
        entrenamiento = entrenamiento.T
        y_train = mapea_clases(y_train[0]).T
        prueba = np.take(x,test_f,axis=0)
        prueba = prueba.T
        y_test = np.take(y,test_f)
        y_test = y_test.tolist()
        print("\nfolder " + str(i))
        i += 1
        modelo = red_neuronal(entrenamiento,y_train,epocas,learning)
        AL, caches = L_model_forward(prueba, modelo)
        Y_pred = np.asarray(np.argmax(AL, axis=0))
        
        precision.append(precision_score(y_test[0], Y_pred[0], average='micro',zero_division=1))
    return np.asarray(precision)

        


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



def actualiza_parametros(cache, grads, taza_aprendizaje=.02):
    parametros = {}
    L = len(cache.items()) // 4

    for l in range(1,L + 1):
        WN = cache["W" + str(l)]
        bN = cache["b" + str(l)]
        dWN = grads["dW" + str(l)]
        dbN = grads["db" + str(l)]

        WN = WN - taza_aprendizaje * dWN
        bN = bN - taza_aprendizaje * dbN
        
        parametros["W" + str(l)] = WN
        parametros["b" + str(l)] = bN
    
    return parametros

def progress_bar(costo, epocas):
    print('{0}>> {1}'.format('='*(costo//(epocas//100)),costo),end="\r")


def formato_orange(dataset,nombre):

    x, y = leeDatos(dataset)
    x = x.T
    y = np.asmatrix(y,dtype=np.int32)
    y = y.T
    atr_name =  ["atr"+str(i) for i in range(x.shape[1])]
    atr_name.append("clase")
    atr_name = np.asmatrix(atr_name)
    atr_type =  ["c" for i in range(x.shape[1])]
    atr_type.append("d")
    atr_type = np.asmatrix(atr_type)
    atr_meta =  ["" for i in range(x.shape[1])]
    atr_meta.append("c")
    atr_meta = np.asmatrix(atr_meta)

    header = np.concatenate((atr_name,atr_type))
    header = np.concatenate((header,atr_meta))
    body = np.concatenate((x,y),axis=1)

    files = np.concatenate((header,body))
    np.savetxt(nombre,files,delimiter=",",fmt="%s")

def red_neuronal(X,Y,epocas,taza_apre):
    parametros = inicializa_parametros([X.shape[0],X.shape[0],Y.shape[0]])
    for i in range(0, epocas +1):
        AL, caches = L_model_forward(X, parametros)
        cost = compute_cost(AL, Y)
        progress_bar(i,epocas)
        grads = backpropagation_NL(caches, X, Y)
        parametros = actualiza_parametros(caches, grads,taza_apre)
    return parametros


"""
Incio del programa principal
"""

@click.command()
@click.option('-c','--capas', default='a', type= str, help='Numero de capas ocultas ej. --capas= 245')
@click.option('-lr','--learning', default=.02, help='Learning rate de la red neuronal')
@click.option('-ep','--epocas', default=500, help='numero de epocas en la red neuronal')
@click.option('-dt','--data',required=True, type=str,  help='Nombre del conjunto de entrenamiento')
@click.option('-pb','--prueba',required=True, type=str, help='Nombre del conjunto de pruebas')
@click.option('-kf','--folds',default=5, help='Numero de folders para la validacion cruzada')
def main(capas, learning, data, prueba, epocas,folds):
    
    """
    Incio de la red neuronal
    """
    X, Y = leeDatos(data)
    Y_tmp = np.asmatrix(Y)
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
    print(X.shape,Y.shape)
    
    
    """
    Entrenamiento de la red neuronal multicapa
    """
    modelo = red_neuronal(X,Y,epocas,learning)
    print("\n")
    
    """
    Evaluando modelo resultante conjunto de prueba
    """
    XP, YP = leeDatos(prueba)
    AL, caches = L_model_forward(XP, modelo)
    Y_pred = np.asarray(np.argmax(AL, axis=0))
    print("Matriz de confucion de MLP\n")
    print(confusion_matrix(YP, Y_pred[0]))
    print("\nPrecision testdata del clasificador MLP")
    print(precision_score(YP, Y_pred[0], average='micro',zero_division=1))
    
    """
    Entrenando modelo con validacion cruzada
    """
    scores = validacionCruzada(X,Y_tmp,folds,epocas,learning)
    print("\nPrecision validacion cruzada de MLP")
    print(np.average(scores))
    

    """
    Incio de Naive Bayes en Orange3
    """
    print("\n########### Naive Bayes #############\n")
    formato_orange(data,'orange.csv')
    formato_orange(prueba,'orangePrueba.csv')
    conjunto_entrena = og.data.Table('orange')
    test = og.data.Table('orangePrueba')
    clasificador = og.classification.NaiveBayesLearner()
    res = og.evaluation.TestOnTestData(conjunto_entrena,test,[clasificador])
    Y = res.actual
    Y_pred = res.predicted

    print("\nMatriz de confucion de Naive B\n")
    print(confusion_matrix(Y, Y_pred[0]))
    tstini = og.evaluation.scoring.Precision(res, average='micro')
    print("\nPrecision testdata de Naive Bayes Orange")
    print(tstini)

    res = og.evaluation.CrossValidation(conjunto_entrena,[clasificador], k = folds)
    print("\nPrecision validacion cruzada de Naive Bayes Orange")
    tstini = og.evaluation.scoring.Precision(res, average='micro')
    print(tstini)

if __name__ == "__main__":
    main()