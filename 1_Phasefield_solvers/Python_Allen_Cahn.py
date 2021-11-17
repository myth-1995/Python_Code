# Importing the required library

import numpy as np
import matplotlib.pyplot as plt

# Define the Physical parameters

h = 1  # Grid size in x and y directions
N = 64  # Grid Dimension in x and y directions
L = N * h  # Boundary Dimension
k = 1  # Relaxation rate
eps = 4  # Gradient Energy
U = 1  # Free Energy Multiplier
a = 0.01  # Initialization parameter

# Defining the arrays representing grid space in x and y directions

x = np.arange(64)  # Defining an array to store grid points in x direction
y = x  # Defining an array to store grid points in x direction

# Defining the time step size and no of time steps

dt = 0.001  # Defining the time step size
nsteps = 5000  # Defining the no of time steps


# Defining the functions to compute the bulk free energy density f(phi)

def f(xi): return np.square(xi - 1) * np.square(xi + 1)


# Defining the function to compute the first derivative of the free
# energy function df/dphi

def f1(xj): return 4 * xj * (np.square(xj) - 1)


# Initializing the phase field distribution

# Filling the phase field array phi with random numbers between -a/2 and a/2

phi = np.random.uniform(low=-a / 2, high=a / 2, size=(N, N))

# Defining arrays for applying the Periodic boundary conditions

plus = np.append(np.arange(1, N), 0)

minus = np.append(N - 1, np.arange(0, N - 1))

# Initializing an array to store the value of free energy functional over time

Fplot = np.arange(nsteps)

# Solving the Allen Cahn phase field equation

for i in range(nsteps):
    # Defining a function to compute numerical approximation of Laplacian of
    # phase field in 2D using Central Difference method

    def laplacephi(phi): return (phi[:, plus] + phi[:, minus] + phi[plus, :] + phi[minus, :] - 4 * phi) / (h * h)


    # Derivative of phase field in x direction : dphi/dx,
    # using Central Difference Method with Periodic Boundary conditions

    def phidx(phi): return (phi[:, plus] - phi[:, minus]) / (2 * h)


    # Derivative of phase field in y direction : dphi/dy,
    # using Central Difference Method with Periodic Boundary conditions

    def phidy(phi): return (phi[plus, :] - phi[minus, :]) / (2 * h)


    # Computing the numerical approximation of free energy functional in 2D

    F = sum(sum(f(phi))) * U + sum(sum(np.square(phidx(phi)) + np.square(phidy(phi)))) * eps

    # Storing the Free energy functional value for plotting

    Fplot[i] = F

    # Computing variation of F with respect to phi : dF/dphi

    Fdphi = f1(phi) * U - 2 * eps * laplacephi(phi)

    # Computing evolution of the phase field using Allen-Cahn equation

    phi = phi - k * dt * Fdphi

# Plotting the final phase field distribution

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
