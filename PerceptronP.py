import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size)  # No need to add +1 for bias here
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else -1

    def predict(self, x):
        z = self.weights.T.dot(x)
        return self.activation_function(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                prediction = self.predict(X[i])
                self.weights += self.learning_rate * (y[i] - prediction) * X[i]

# Datos de entrenamiento 
#   X0  X1  X2
X = np.array([
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
])
y = np.array([1, 1, 1, -1])

# Entrenamiento
perceptron = Perceptron(input_size=3)
perceptron.fit(X, y)

# Mostrar Resultados
for x in X:
    print(f"Entrada: {x}, Predicción: {perceptron.predict(x)}")
