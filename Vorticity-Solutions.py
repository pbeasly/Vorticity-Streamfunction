import numpy as np
import scipy.sparse
import scipy.integrate
import matplotlib.pyplot as plt
import pdb

# Define function for Matrix A

def A(N):
    e1 = np.ones(N)  # vector of ones
    e2 = -1 * e1  # vector of -1s
    A = scipy.sparse.spdiags([e1, e2, e1, e2], [-(N - 1), -1, 1, N - 1], N, N)

    return A

# Define constants for problem 1
L = 10
tend = 10
t_step = 0.5
tvals = np.arange(0, tend+t_step, t_step)
x = np.linspace(-L, L, 201)
x = x[:-1]
N = len(x) # N value in x and y directions
h = 2*L/N
A = (1/(2*h))*A(N)  # create matrix from function


#plt.spy(A, markersize='8', marker='o') # view the matrix structure
#plt.show() # check matrix structure

# Define initial conditions and functions
f = lambda x: np.exp(-1*(x-5)**2)  # initial condition
y0 = f(x)

def advPDE(t, u, A):  # PDE function
    u_t = 0.5*A@u
    return u_t

# pass into ivp solver
sol = scipy.integrate.solve_ivp(lambda t, u: advPDE(t, u, A), [0, tend], y0, t_eval=tvals)


# part C)______

# Define c(t), deal with heaviside function seperately
c = lambda t: 1+2*np.sin(5*t)

# determine index for heavyside function
xsplit = int((4 + L) / h)  # index for x = 4

H = np.ones([200, 200]) # initialize heaviside matrix

# place ones in all values for x<= 4
for i in range(len(x)):
    H[:xsplit, :] = 0

A = A.todense() # for applying heaviside function

HA = np.multiply(H, A) # element-wise multiplication

# initial condition
y0 = f(x)

def advPDE2(t, u):  # PDE function
    matrix_A = np.add(c(t)*A, -1*HA)
    u_t = matrix_A@u.T
    return u_t

# pass into ivp solver
sol = scipy.integrate.solve_ivp(lambda t, u: advPDE2(t, u), [0, tend], y0, t_eval=tvals)


# Problem 2
# Define Matrix Functions____
# Define Laplacian
def LAP(m):
    n = m*m # total size of matrix

    e1 = np.ones(n)  # vector of ones
    Low1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,)) # Lower diagonal 1
    Low2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,)) #Lower diagonal 2
                                        # Low2 is NOT on the second lower diagonal,
                                        # it is just the next lower diagonal we see
                                        # in the matrix.

    Up1 = np.roll(Low1, 1)  # Shift the array for spdiags
    Up2 = np.roll(Low2, m-1)  # Shift the other array

    LAP = scipy.sparse.spdiags([e1, e1, Low2, Low1, -4*e1, Up1, Up2, e1, e1],
                             [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)], n, n, format='csc')
    LAP[0,0]=2
    return LAP

#plt.spy(LAP_func(6), markersize='8', marker='o') # view the matrix structure
#plt.show()

# Need to function for B matrix
def B(m):
    n = m*m  # total size of matrix
    e0 = np.ones(n)  # vector of ones

    # Create matrix from left-bottom to right-top
    B = scipy.sparse.spdiags([e0, -1*e0, e0, -1*e0],
                             [-1*(m**2-m), -m, m, m**2-m], n, n, format='csc')
    return B

# Create function for C matrix
# First create a function to generate required diagonals for C matrix
# Build from bottom-left to top-right; e1, e2, e3, e4
def C_Diags(m):

    # Algorithm to create m sets of [1,0,0,0...] diag
    e0 = [1]
    ex = np.zeros(m - 1)
    e0 = np.append(e0, ex)
    e1 = []
    for i in range(m):
        e1 = np.append(e1, e0)

    # Algorithm to create m sets of [-1,-1,-1...,0] diag
    e0 = -1*np.ones(m-1)
    e0 = np.append(e0, 0)
    e2 = []
    for i in range(m):
        e2 = np.append(e2, e0)

    # Algorithm to create m sets of [1,1,1...,0] diag
    e0 = np.ones(m-1)
    e0 = np.append(e0, 0)
    e3 = []
    for i in range(m):
        e3 = np.append(e3, e0)

    # Algorithm to create m sets of [-1,0,0,0...] diag
    e0 = [-1]
    ex = np.zeros(m-1)
    e0 = np.append(e0, ex)
    e4 = []
    for i in range(m):
        e4 = np.append(e4, e0)

    return e1, e2, e3, e4

# Create the C matrix
def C(m):
    n = m*m
    e1, e2, e3, e4 = C_Diags(m)
    e3 = np.roll(e3, 1)
    e4 = np.roll(e4, -1)
    # Create matrix from left-bottom to right-top
    C = scipy.sparse.spdiags([e1, e2, e3, e4],
                             [-(m-1), -1, 1, m-1], n, n, format='csc')
    return C

# Check matrix structure
# print(LAP(4).todense())
# print(np.shape((LAP(4))))

# Define Constants_____
L = 10
tend = 4
dt = 0.5
tspan = np.arange(0, tend+dt, dt)
x = np.linspace(-L, L, 65)
x = x[:-1]
y = np.linspace(-L, L, 65)
y = y[:-1]
h = 2*L/64 # step size in x & y
mu = 0.001
n = len(x) # matrix size for x & y

# initial condition for w
f3 = lambda x, y: np.exp(-2*x**2-(y**2/20))

w0 = np.zeros([64, 64]) # initialize initial column vector

for i in range(len(x)):
    for j in range(len(y)):
        w0[i,j] = f3(x[i], y[j])

w0 = w0.reshape([-1,1]) # reshape to a column vector
y0 = np.squeeze(w0) # make one dimenstional for the solver


# Generate Matrices from functions above
A = (1/(h**2))*LAP(n)
B = (1/(2*h))*B(n)
C = (1/(2*h))*C(n)

# assign deliveraables
A4 = A.todense()
A5 = B.todense()
A6 = C.todense()


# Define ODE
# dwd/t = (C@phi)@(B@w)-(B@phi)@(C@w)+mu*(A@w)
# and A*phi = w

#phi = scipy.sparse.linalg.spsolve(A, w0)
# w_t = (C@phi)@(B@w0)-(B@phi)@(C@w0)+mu*(A@w0)

def vortPDE(t, w):
    phi = scipy.sparse.linalg.spsolve(A, w) # this needs to be the GE method and PLU
    w_t = (C@phi)*(B@w)-(B@phi)*(C@w)+mu*(A@w)
    return w_t

sol3 = scipy.integrate.solve_ivp(lambda t, w: vortPDE(t, w), [0, tend], y0, t_eval=tspan)

A7 = sol3.y.T  # assign to deliverable

# for PLU solution change A to PLU
PLU = scipy.sparse.linalg.splu(A)

def vortPDE(t, w):
    phi = PLU.solve(w)
    w_t = (C@phi)*(B@w)-(B@phi)*(C@w)+mu*(A@w)
    return w_t

sol4 = scipy.integrate.solve_ivp(lambda t, w: vortPDE(t, w), [0, tend], y0, t_eval=tspan)





