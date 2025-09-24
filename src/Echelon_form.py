# Row Echelon Form (Matrix input from user)

def row_echelon_form(A):۳۳
    A = [row[:] for row in A]
    rows = len(A)
    cols = len(A[0])
    pivot_row = 0  # index of current row

    for col in range(cols):
        # Find pivot in or below the current row
        pivot = None
        for r in range(pivot_row, rows):
            if A[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue  # no pivot in this column

        # Swap current row with pivot row if needed
        if pivot != pivot_row:
            A[pivot_row], A[pivot] = A[pivot], A[pivot_row]

        # Normalize pivot row to make pivot = 1
        pivot_val = A[pivot_row][col]
        A[pivot_row] = [x / pivot_val for x in A[pivot_row]]

        # Eliminate all rows below pivot
        for r in range(pivot_row + 1, rows):
            factor = A[r][col]
            if factor != 0:
                A[r] = [a - factor * b for a, b in zip(A[r], A[pivot_row])]

        pivot_row += 1
        if pivot_row == rows:
            break

    return A


# ---- Get matrix from user ----
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

A = []
print("Enter the matrix row by row (separate numbers by space):")
for i in range(rows):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    if len(row) != cols:
        raise ValueError("Number of entries in row does not match columns")
    A.append(row)

# Compute row echelon form
ref = row_echelon_form(A)

print("\nRow Echelon Form:")
for row in ref:
    print(row)

