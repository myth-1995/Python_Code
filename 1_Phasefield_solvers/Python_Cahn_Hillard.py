import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

h = 1  # Grid size in x and y directions
N = 64  # Grid Dimension in x and y directions
L = N * h  # Boundary Dimension
eps = 4  # Gradient Energy
U = 1  # Free Energy Multiplier
a = 0.01  # Initialization parameter
M = 1  # Diffusion Coefficient
delta = 10 ** -8  # Tolerance

x = np.arange(64)  # Defining an array to store grid points in x direction
y = x  # Defining an array to store grid points in x direction

dt = 0.001  # Defining the time step size
nsteps = 30000  # Defining the no of time steps


# Defining the functions to compute the bulk free energy density

def f(xi): return np.square(xi - 1) * np.square(xi + 1)


# Defining the function to compute the first derivative of the free energy function

def f1(xj): return 4 * xj * (np.square(xj) - 1)


# Defining the function to compute second derivative of free energy function phi

def f2(xk): return 12 * np.square(xk) - 4


# Defining a function to compute third derivative of free energy function w.r.t phi

def f3(xl): return 24 * xl


# Initializing the phase field distribution
# Filling the phase field array phi with random numbers between -a/2 and a/2

phi = np.random.uniform(low=-a / 2, high=a / 2, size=(N, N))
phi1 = np.random.uniform(low=-a + 1 / 2, high=a + 1 / 2, size=(N, N))

# Defining arrays for applying the Periodic boundary conditions

plus = np.append(np.arange(1, N), 0)
minus = np.append(N - 1, np.arange(0, N - 1))

# Initializing an array to store the value of free energy functional over time

Fplot = np.arange(nsteps)

# Solving the Cahn-Hillard phase field equation

for i in range(nsteps):

    # Defining a function to compute numerical approximation of Laplacian of
    # phase field in 2D using Central Difference method

    def lphi(phi):
        return (phi[:, plus] + phi[:, minus] + phi[plus, :] + phi[minus, :] - 4 * phi) / (h * h)


    #  Defining a function to compute numerical approximation of Laplacian
    #  of Laplace using Central Difference Method in 2D

    def llphi(phi):
        return (lphi(phi)[:, plus] + lphi(phi)[:, minus] + lphi(phi)[plus, :] + lphi(phi)[minus, :] - 4 * lphi(phi)) / (
                h * h)


    # Derivative of phase field in x direction : dphi/dx,
    # using Central Difference Method with periodic boundary conditions

    def phidx(phi):
        return (phi[:, plus] - phi[:, minus]) / (2 * h)


    # Derivative of phase field in y direction : dphi/dy,
    # using Central Difference Method with periodic boundary conditions

    def phidy(phi):
        return (phi[plus, :] - phi[minus, :]) / (2 * h)


    # Computing the numerical approximation of free energy functional in 2D

    F = sum(sum(f(phi))) * U + sum(sum(np.square(phidx(phi)) + np.square(phidy(phi)))) * eps

    # Storing the Free energy functional value for plotting

    Fplot[i] = F

    # Computing the numerical approximation of g[phi]

    Gdphi = np.multiply(f2(phi), lphi(phi)) * U + np.multiply(f3(phi), (
            np.square(phidx(phi)) + np.square(phidy(phi)))) * U - 2 * eps * llphi(phi)

    # Computing evolution of the phase field using Cahn-Hillard equation

    # Uncomment this line and comment the lines below this line to run the Explicit euler method

    # phi = phi + M * dt * Gdphi

    # Computing the predictor STEP: 1

    phi0 = phi + M * dt * Gdphi

    # etting the initial value for error

    error = 1

    # Implicit Euler Method

    while LA.norm(error) > delta:
        # Computing g[phi_0] to compute the corrector STEP: 2

        Gdphi0 = np.multiply(f2(phi0), lphi(phi0)) * U + np.multiply(f3(phi0), (
                np.square(phidx(phi0)) + np.square(phidy(phi0)))) * U - 2 * eps * llphi(phi0)

        # Computing phi_1 corrector STEP: 2

        phi1 = phi + (Gdphi + Gdphi0) * (dt / 2)

        # Computing the error STEP: 3

        error = phi1 - phi0

        # Updating the phase field value STEP: 4

        phi0 = phi1

    # Computing the phase field in new time step

    phi = phi1

# Plotting the Final Phase Field Distribuition

plt.close('all')

fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])

X, Y = np.meshgrid(x, y)

cp = ax.contour(X, Y, phi)
ax.clabel(cp, inline=True,
          fontsize=10)
ax.set_title('Final phase field distribution')
ax.set_xlabel('x ')
ax.set_ylabel('y ')
plt.show()

# Plotting the Free energy functional over time

plt.close('all')

t = np.arange(nsteps)
plt.plot(t / 1000, Fplot / 1000)
plt.title("Free energy Functional vs time")
plt.xlabel("time")
plt.ylabel("F/10^3")
plt.show()
