import sys
import numpy as np

# def next_power_of_2(x):
#     """Find the next power of 2 greater than or equal to x."""
#     return 1 if x == 0 else 2**(x - 1).bit_length()

def next_even_size(x):
    """Find the next even number greater than or equal to x."""
    return x if x % 2 == 0 else x + 1

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

# Standard Matrix Multiplication 
def standard_matrix_multiplication(X, Y):
    n = X.shape[0]
    Z = np.zeros((n, n))  
    
    for i in range(n):
        for j in range(n):
            Z[i, j] = np.sum(X[i, :] * Y[:, j])
    return Z

# Strassen Matrix Multiplication 
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


def read_matrix_from_file(file_path, d):
    """Read two matrices of size dxd from a file."""
    with open(file_path, 'r') as file:
        numbers = [int(line.strip()) for line in file.readlines()]
    
    A = np.array(numbers[:d**2]).reshape(d, d)
    B = np.array(numbers[d**2:]).reshape(d, d)
    
    return A, B

def main(f, d, i):
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: ./strassen <flag> <dimension> <inputfile>")
        sys.exit(1)

    flag = f
    dimension = d
    input_file = i
    
    # Read matrices from the input file
    A, B = read_matrix_from_file(input_file, dimension)
    
    # Perform Strassen's matrix multiplication
    C = strassen_matrix_multiplication(A, B, 38)  # Example threshold n_0=2, you might want to adjust this based on your needs
    
    # Print the diagonal elements of the result
    for i in range(dimension):
        print(int(C[i, i]))

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
