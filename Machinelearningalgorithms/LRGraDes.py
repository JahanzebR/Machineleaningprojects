import numpy as np


# Let m = #training examples, n = number of features
# the following: y is R^(1 x m), X is R^(n x m), w is R^(n x 1)


class LinearRegression():
    def __init__(self):
        self.learning_rate = 0.01
        self.total_iterations = 10000

    def y_hat(self, X, w):  # yhat = w1x1 + w2x2 + x3x3 +...
        return np.dot(w.T, X)

    def loss(self, yhat, y):
        L = 1 / self.m * np.sum(np.power(yhat - y, 2))
        return L

    def gradient_descent(self, w, X, y, yhat):
        # (1 x m), # (n x m)
        # n x 1
        dldW = 2 / self.m * np.dot(X, (yhat - y).T)

        w = w - self.learning_rate * dldW
        return w

    def main(self, X, y):
        x1 = np.ones((1, X.shape[1]))
        X = np.append(X, x1, axis=0)

        self.m = X.shape[1]
        self.n = X.shape[0]

        w = np.zeros((self.n, 1))

        for it in range(self.total_iterations + 1):
            yhat = self.y_hat(X, w)
            loss = self.loss(yhat, y)

            if it % 2000 == 0:
                print(f'Cost at iteration {it} is {loss}')

            w = self.gradient_descent(w, X, y, yhat)

        return w


if __name__ == "__main__":
    X = np.random.rand(1, 500)  # y= w1 + w2x2
    y = 3 * X + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.main(X, y)
