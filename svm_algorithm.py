import numpy as np
def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> (list, float):
	
    w = np.zeros(len(labels))
    b = 0
    t = 0
    gamma = 1 / (2 * sigma * sigma)
    
    def ker(x, y, kernel):
        if kernel == 'linear':
            return np.dot(x, y)
        elif kernel == 'rbf':
            dist_sq = np.sum((x - y)**2)
            val = np.exp(-gamma * dist_sq) 
            return val
    
    for i in range(0, iterations):
        t += 1
        for j in range(0, len(labels)):
            
            curr_x = data[j]
            sumation = 0
            for k in range(0, len(labels)):
                sumation += w[k] * labels[k] * ker(curr_x, data[k],kernel)
            pred = sumation + b
            if labels[j]*pred < 1:
                nt= 1 / (lambda_val * t)
                w[j]+=nt*(labels[j]-lambda_val*w[j])
                b += labels[j] * nt
            
    return w,b
