import numpy as np

def check_matrix_properties():
    """
    Ask the user for a matrix (rows x cols) and then
    check linear dependence, independence, and spanning
    based on the rank.
    """
    # --- Get dimensions of the matrix ---
    n = int(input("Enter number of rows n (dimension of space R^n): "))
    k = int(input("Enter number of columns k (number of vectors): "))

    # --- Prompt the user to enter the matrix entries row by row ---
    print("Enter the matrix entries row by row (space separated):")
    A = []
    for i in range(n):
        # Ask for each row and split into float numbers
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != k:
            raise ValueError("Each row must have exactly k entries.")
        A.append(row)
    A = np.array(A)

    # --- Compute the rank of the matrix ---
    rank = np.linalg.matrix_rank(A)

    # --- Display the matrix and its rank ---
    print("\nMatrix A:")
    print(A)
    print("Rank(A):", rank)

    # --- Check linear dependence / independence ---
    if k > n:
        # More vectors than the dimension => automatically dependent
        print("\nBy lemma: k > n, so the vectors are linearly dependent.")
    else:
        if rank < k:
            # Rank less than number of vectors => dependent
            print("\nThe vectors are linearly dependent (rank < k).")
        else:
            # Full rank equals number of vectors => independent
            print("\nThe vectors are linearly independent (rank = k).")

    # --- Check whether the vectors span R^n ---
    if rank == n:
        print("The vectors span R^n (rank = n).")
    else:
        print("The vectors do NOT span R^n (rank < n).")

    # --- Optionally test whether a given vector b lies in the span ---
    ans = input("\nDo you want to test if a vector b is in the span? (y/n): ")
    if ans.lower() == 'y':
        # Ask the user to enter vector b
        b = list(map(float, input(f"Enter vector b of length {n} (space separated): ").split()))
        if len(b) != n:
            raise ValueError("Vector b must have length n.")
        b = np.array(b)

        # Solve A c = b in least squares sense
        c, residuals, rank2, s = np.linalg.lstsq(A, b, rcond=None)

        # Check if the computed combination exactly reproduces b
        if np.allclose(A @ c, b):
            print("b is in the span of the given vectors.")
            print("One possible combination c:", c)
        else:
            print("b is NOT in the span of the given vectors.")

# Run the function:
check_matrix_properties()

