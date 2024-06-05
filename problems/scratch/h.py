import numpy as np
import matplotlib.pyplot as plt

N = 500

h = np.ones(N)
w = 0.5
alpha = 1.0
max_iterations = 1000

ui_grid = np.arange(0.5, N, 1.0) / float(N)

for iteration in range(0, max_iterations):
    h_prev = h.copy()
    h_new = np.zeros_like(h)
    
    for i in range(1, N):
        ui = (i - 0.5) / float(N)
        integral_operator = 0
        for j in range(1, N):
            uj = (j - 0.5) / float(N)
            integral_operator += (ui * h[i]) / (ui + uj)
        final_term = 1.0 / (1.0 - (w / 2.0 / float(N)) * integral_operator)    

        h_new[i] = h[i] - final_term

    res = np.sqrt((np.power((h_new - h_prev), 2.0) / float(N)).sum())

    print(iteration, res)

    h = alpha * h_new + (1.0 - alpha) * h_prev
    

plt.plot(ui_grid, h)
plt.show()
