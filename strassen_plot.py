import numpy as np
import matplotlib.pyplot as plt
import time

def pad_matrix(A):
    rows, cols = A.shape
    # Only pad if not even
    if rows % 2 == 1:
        A = np.pad(A, ((0, 1), (0, 0)), mode='constant')
    if cols % 2 == 1:
        A = np.pad(A, ((0, 0), (0, 1)), mode='constant')
    return A

def remove_padding(A, original_shape):
    return A[:original_shape[0], :original_shape[1]]

def standard_matrix_multiplication(X, Y):
    n, m = X.shape[0], Y.shape[1]
    Z = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Z[i, j] = np.dot(X[i, :], Y[:, j])
    return Z

def strassen_matrix_multiplication(X, Y, n_0):
    # Base case: use standard multiplication for small matrices
    if X.shape[0] <= n_0 or X.shape[1] <= n_0 or Y.shape[1] <= n_0:
        return standard_matrix_multiplication(X, Y)

    # Ensure matrices are square and dimensions are even
    original_shape_X = X.shape
    original_shape_Y = Y.shape

    X = pad_matrix(X)
    Y = pad_matrix(Y)
    
    n = X.shape[0]

    # Split matrices into quarters
    mid = n // 2
    A, B, C, D = X[:mid, :mid], X[:mid, mid:], X[mid:, :mid], X[mid:, mid:]
    E, F, G, H = Y[:mid, :mid], Y[:mid, mid:], Y[mid:, :mid], Y[mid:, mid:]

    # 7 recursive multiplications
    P1 = strassen_matrix_multiplication(A, F - H, n_0)
    P2 = strassen_matrix_multiplication(A + B, H, n_0)
    P3 = strassen_matrix_multiplication(C + D, E, n_0)
    P4 = strassen_matrix_multiplication(D, G - E, n_0)
    P5 = strassen_matrix_multiplication(A + D, E + H, n_0)
    P6 = strassen_matrix_multiplication(B - D, G + H, n_0)
    P7 = strassen_matrix_multiplication(A - C, E + F, n_0)

    # Combine the results
    Z_top_left = P5 + P4 - P2 + P6
    Z_top_right = P1 + P2
    Z_bottom_left = P3 + P4
    Z_bottom_right = P1 + P5 - P3 - P7

    # Assemble the final matrix and remove any extra padding
    Z = np.vstack((np.hstack((Z_top_left, Z_top_right)), np.hstack((Z_bottom_left, Z_bottom_right))))
    Z = remove_padding(Z, (original_shape_X[0], original_shape_Y[1]))
    return Z

n_0_even_sparse = range(10, 200, 2)
n_0_odd_sparse = range(11, 200, 2)

# Initialize lists for the sparser runtimes
strassen_runtimes_even_sparse = []
standard_runtimes_even_sparse = []
strassen_runtimes_odd_sparse = []
standard_runtimes_odd_sparse = []

# Compute runtimes for the sparser even n_0 values
for n_0 in n_0_even_sparse:
    X = np.random.randint(0,2, (n_0, n_0))
    Y = np.random.randint(0,2, (n_0, n_0))

    a = ((n_0/2) + 1)

    start_time = time.time()
    standard_matrix_multiplication(X, Y)
    standard_runtimes_even_sparse.append(time.time() - start_time)

    start_time = time.time()
    strassen_matrix_multiplication(X, Y, a)
    strassen_runtimes_even_sparse.append(time.time() - start_time)

# Compute runtimes for the sparser odd n_0 values
for n_0 in n_0_odd_sparse:
    X = np.random.randint(0,2, (n_0, n_0))
    Y = np.random.randint(0,2, (n_0, n_0))
    a = ((n_0/2) + 1)

    start_time = time.time()
    standard_matrix_multiplication(X, Y)
    standard_runtimes_odd_sparse.append(time.time() - start_time)

    start_time = time.time()
    strassen_matrix_multiplication(X, Y, a)
    strassen_runtimes_odd_sparse.append(time.time() - start_time)

# Plotting the results for the sparser even and odd n_0 values
plt.figure(figsize=(12, 14))

# Plot for even n_0 values
plt.subplot(2, 1, 1)
plt.plot(n_0_even_sparse, strassen_runtimes_even_sparse, marker='o', label="Strassen's Algorithm (Even $n_0$)")
plt.plot(n_0_even_sparse, standard_runtimes_even_sparse, marker='x', label='Standard Multiplication (Even $n_0$)')
plt.title('Runtime for Even $n_0$ Values')
plt.xlabel('$n_0$ Value')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
plt.legend()

# Plot for odd n_0 values
plt.subplot(2, 1, 2)
plt.plot(n_0_odd_sparse, strassen_runtimes_odd_sparse, marker='o', label="Strassen's Algorithm (Odd $n_0$)")
plt.plot(n_0_odd_sparse, standard_runtimes_odd_sparse, marker='x', label='Standard Multiplication (Odd $n_0$)')
plt.title('Runtime for Odd $n_0$ Values')
plt.xlabel('$n_0$ Value')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
