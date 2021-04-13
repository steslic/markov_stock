import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange two rows of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    reduce row j using row i with pivot pivot, in matrix A
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        # we allow an accumulation of error 100 times larger 
        # than a single computation
        # this is crude but works for computations without a large 
        # dynamic range
        if relError(A[j][k], factor * A[i][k]) < 100 * np.finfo('float').resolution:
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]

# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy()
    m, n = np.shape(A)
    for i in range(m-1):
        # Let lefmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            swapRows(A, leftmostNonZeroRow, i)
        # Use row reduction operations to create zeros in all positions 
        # below the pivot.
        for h in range(i+1,m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

#################### 

# If any operation creates a row that is all zeros except the last element,
# the system is inconsistent; stop.
def inconsistentSystem(B):
    """
    B is assumed to be in echelon form; return True if it represents
    an inconsistent system, and False otherwise
    """
    rows = len(B)
    cols = len(B[0])
    
    counter = 0 
    
    if B[rows - 1][cols - 1] != 0:
        for i in range(0, cols - 1):
            if B[rows - 1][i] == 0:
                counter += 1
                
                if counter == cols - 1:
                    return True # inconsistent
                    
            
    return False # consistent
    
    

def backsubstitution(B):
    """
    return the reduced row echelon form matrix of B
    """
    rows = len(B)
    cols = len(B[0])
    
    zrows = 0 
    
    # check for zero rows
    for i in range(rows - 1, -1, -1):
        tv = False
        for j in range(cols):
            if B[i][j] != 0:
                tv = True
                break
            
        if tv == False:
            zrows += 1
            
    # divide each row by pivot 
    for i in range(rows - zrows):
        for j in range(i + 1, cols):
            B[i][j] = B[i][j] / B[i][i]
        
        B[i][i] = B[i][i] / B[i][i] # get 1 
        
        
    # row reduction
    for i in range(rows - zrows):
        for j in range(cols):
            if B[i][j] != 0:
                pivot = j
                break
            
        for k in range(i - 1, -1, -1):
            rowReduce(B, i, k, pivot)
    
    return B