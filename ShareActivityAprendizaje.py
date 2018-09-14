import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
#Cristian Geovanni Aguilar Valencia A01066467

#Neurona realizada en clase con cambios en la variable hidden para obtener una mejor aproximación
class RedNeuronal(object):
    def __init__(self, Lambda = 0):
        self.inputs = 8
        self.outputs = 1
        self.hidden = 30 # Los mejores resultados se obtuvieron con: 1,5,10,30
        self.W1 = np.random.randn(self.inputs, self.hidden)
        self.W2 = np.random.randn(self.hidden, self.outputs)
        self.Lambda = Lambda
    def sigmoide(self,z):
        return 1/(1+ np.exp(-z))
    def feedForward(self,x):
        self.z2 =  x @ self.W1
        self.a2 =  self.sigmoide(self.z2)
        self.z3 =self.a2 @ self.W2
        self.yhat =  self.sigmoide(self.z3)
        return self.yhat
    def sigmoideDerivada(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    def funcionCosto(self,x,y):
        self.yhat = self.feedForward(x)
        Costo = 0.5*sum((y-self.yhat)**2)/x.shape[0] + (self.Lambda/2) * (np.sum(self.W1**2) +np.sum(self.W2**2))
        return Costo
    def funcionDeCostoDerivada(self,x,y):
        self.yhat = self.feedForward(x)
        self.delta3 = np.multiply(-(y - self.yhat ),self.sigmoideDerivada(self.z3))
        djW2 = (np.transpose(self.a2)@self.delta3) / x.shape[0] + (self.Lambda*self.W2)
        self.delta2 = self.delta3@ djW2.T*self.sigmoideDerivada(self.z2)
        djW1 =   (x.T @ self.delta2) / x.shape[0] + (self.Lambda*self.W1)
        return djW1, djW2
    def getPesos(self):
        data = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return data
    def setPesos(self, datos):
        W1_inicio = 0
        W1_fin = self.hidden * self.inputs
        self.W1 = np.reshape(datos[W1_inicio:W1_fin], (self.inputs, self.hidden))
        W2_fin = W1_fin + self.hidden * self.outputs
        self.W2 = np.reshape(datos[W1_fin:W2_fin], (self.hidden, self.outputs))
    def getGradientes(self, X, y):
        djW1, djW2 = self.funcionDeCostoDerivada(X, y)
        return np.concatenate((djW1.ravel(), djW2.ravel()))

class Entrenador:
    def __init__(self, unaRed):
        # referencia a una red local
        self.NN = unaRed
    def actualizaPesos(self, params):
        self.NN.setPesos(params)
        self.Costos.append(self.NN.funcionCosto(self.X, self.y))
        self.CostosTest.append(self.NN.funcionCosto(self.Xtest, self.ytest))
    def obtenPesosNN(self, params, X, y):
        self.NN.setPesos(params)
        cost = self.NN.funcionCosto(X, y)
        grad = self.NN.getGradientes(X, y)
        return cost, grad
    def entrena(self, Xtrain, ytrain, Xtest, ytest):
        # variables para funciones callback
        self.X = Xtrain
        self.y = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        # lista temporal de costos
        self.Costos = []
        self.CostosTest = []
        pesos = self.NN.getPesos()
        opciones = {'maxiter': 200, 'disp': True}
        # self.obtenPesosNN, funcion objetivo
        # args=(X, y), input / output data
        # salida: regresa el costo y los gradientes
        salida = optimize.minimize(self.obtenPesosNN, pesos, jac=True, method='BFGS', args=(Xtrain, ytrain), options=opciones, callback=self.actualizaPesos)
        self.NN.setPesos(salida.x)
        self.resultados = salida

#Utilizando pandas se trabaja con el dataset
#Se utilizaron todas las variables del dataset ya que todas son variables relevantes para saber si una persona es propensa a tener diabetes
columns = ['pregnant','glucose','blood pressure','skin fold thickness','serum insulin','mass index','pedigree function','Age']
#El objetivo es predecir si puedes tener diabetes o no, por lo que se debe utilizar la columna Diabetes como target ya que es la que indica si la persona tiene o no diabetes
target = ['Diabetes']
'''Este es el dataset con la información para realizar la predicción, contiene datos del National Institute of Diabetes and Digestive and Kidney Diseases
de mujeres mayores a 21 años con 768 registros los cuales se van a utilizar para realizar la predicción.'''
diabetes = pd.read_csv("pima-indians-diabetes.csv")
diabetes.head()
#Se divide el dataset en entradas(X) y la salida(Y)
x = diabetes.loc[:, columns]
y = diabetes.loc[:, target]

#Se divide el dataset en dos partes, train y test
#Son 768 registros, el 80% son 614, se utiliza 80% para obtener mejores resultados.
xtrain = x[1:614]
xtest = x[614:]
ytrain = y[1:614]
ytest = y[614:]
#Se normalizan los datos
xtrain = xtrain/np.amax(xtrain,axis=0)
ytrain = ytrain/np.amax(ytrain,axis=0)
xtest = xtest/np.amax(xtest,axis=0)
ytest = ytest/np.amax(ytest,axis=0)

xtrainMatrix = xtrain.as_matrix()
ytrainMatrix = ytrain.as_matrix()
xtestMatrix = xtest.as_matrix()
ytestMatrix = ytest.as_matrix()

#Finalmente se utiliza la red neuronal para hacer la predicción.
rn = RedNeuronal(Lambda=0.0001)
e = Entrenador(rn)
#Se entrena la neurona
e.entrena(xtrainMatrix, ytrainMatrix, xtestMatrix, ytestMatrix)
#Se muestran los resultados
plt.plot(e.Costos)
plt.plot(e.CostosTest)
plt.grid(2)
plt.ylabel("diabetes")
plt.xlabel("iteraciones")
plt.show()
#Conclusión
#Es una buena aproximacion ya que en la grafica se puede observar que la predicción es muy cercana al resultado real,
#sin embargo si fuera un dataset mas grande podría ser aún mejor la aproximación al igual que cambiando el número de hidden en la neurona


