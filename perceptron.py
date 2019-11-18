import numpy as np


class Perceptron(object):
    # no_of_inputs => numero de entradas
    # threshold => numero de iteraciones
    # learning_rate => la tasa de aprendizaje
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        # Crea un array con el tamaño de los input + 1 lleno de 0
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        # Numpy nos permite gestionar el producto escalar con np.dot()
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # el el siguiente bloque gestionamos la variable de activacion y retornamos
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    # Definimos el metodo de entrenamiento.
    # training_inputs => entradas
    # labels => salidas deseadas
    def train(self, training_inputs, labels):
        # Iteramos las veces que indique la variable threshold
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels): # Nos retorna un objeto iterable y lo recorremos
                prediction = self.predict(inputs)  # evaluamos el objecto  obtenido para iterar
                # Aplicamos la formular para calcular los nuevos pesos
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
