import numpy as np

# --- Step 1: Get vectors from user ---
# Each vector should be entered as space-separated numbers (e.g., "1 2 -1")

w1 = np.array(list(map(float, input("Enter vector w1 (e.g. 1 2 -1): ").split())))
w2 = np.array(list(map(float, input("Enter vector w2 (e.g. 2 -3 -1): ").split())))
b = np.array(list(map(float, input("Enter vector b (e.g. 1 0 0): ").split())))

# --- Step 2: Compute the Gram matrix K ---
# K_ij = <w_i, w_j> = dot product between w_i and w_j
K = np.array([
    [np.dot(w1, w1), np.dot(w1, w2)],
    [np.dot(w2, w1), np.dot(w2, w2)]
])

# --- Step 3: Compute the vector f ---
# f_i = <w_i, b>
f = np.array([np.dot(w1, b), np.dot(w2, b)])

# --- Step 4: Solve for x* ---
# Solve the linear system Kx = f
x_star = np.linalg.solve(K, f)

# --- Step 5: Compute the closest point w* ---
# w* = x1*w1 + x2*w2 = W @ x*
W = np.column_stack((w1, w2))  # make matrix with w1 and w2 as columns
w_star = W @ x_star

# --- Step 6: Compute the distance ---
# Distance between b and its projection w*
distance = np.linalg.norm(w_star - b)

# --- Step 7: Show results ---
print("\nGram Matrix (K):\n", K)
print("\nVector f:\n", f)
print("\nSolution x*:\n", x_star)
print("\nClosest Point w*:\n", w_star)
print("\nMinimum Distance ||w* - b|| = ", distance)

