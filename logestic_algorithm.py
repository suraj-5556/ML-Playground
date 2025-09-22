import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0, lr=0.01):
    n = len(labels)
    alpha = np.zeros(n)   # coefficients
    b = 0
    gamma = 1 / (2 * sigma * sigma)
    weights_array = []
    loss_array = []

    # Kernel
    def ker(x, y, kernel):
        if kernel == 'linear':
            return np.dot(x, y)
        elif kernel == 'rbf':
            dist_sq = np.sum((x - y) ** 2)
            return np.exp(-gamma * dist_sq)

    # Prediction (sigmoid)
    def feed_forward(x, alpha, b):
        pred = 0
        for k in range(n):
            pred += alpha[k] * labels[k] * ker(x, data[k], kernel)
        pred += b
        return 1 / (1 + np.exp(-pred))

    # BCE Loss
    def loss(x, y, alpha, b):
        count = 0
        for i in range(len(y)):
            y_pred = feed_forward(x[i], alpha, b)
            y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
            count = count - (y[i] * np.log(y_pred) + (1 - y[i]) * np.log(1 - y_pred))
        return count / len(y)

    # Training (logistic regression style updates)
    for it in range(iterations):
        for j in range(n):
            y_pred = feed_forward(data[j], alpha, b)
            error = labels[j] - y_pred  # gradient error

            # gradient update
            for k in range(n):
                alpha[k] += lr * error * ker(data[j], data[k], kernel)
            b += lr * error

        weights_array.append([np.round(alpha.copy(), 3), round(b, 3)])
        loss_array.append(loss(data, labels, alpha, b))

    return alpha, b, weights_array, loss_array
