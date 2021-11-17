# ----------------------------------------------------------------------------------------------------------------------
#
# Description: The following code solves the boundary value problem, thermo mechanical static analysis of steel 
#              clading
#
# Author:      V Mithlesh Kumar, vmithleshkumar@gmail.com
#
# Reference:  Sascha Maassen, Finite Element Method - Coupled Problems code
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import PyFEMP
import Q1_LE_Dyn_T_template_static as ELEMENT

FEM = PyFEMP.FEM_Simulation(ELEMENT)
n = 8                                                                        
XI, Elem = PyFEMP.msh_conv_quad([-0.01, 0.0], [-0.005, 0.01], [0.005, 0.01], [0.01, 0.0], [3*n, n], type='Q1')
FEM.Add_Mesh(XI, Elem)
# "E", "nu", "rho", "eta", "bx", "by", "a_q", "c", "r", "T0", "a_T"
FEM.Add_Material([210e09, 0.3, 7850, 2000, 0.0, 0.0, 50.0, 460.0, 0.0, 290.15, 7e-6], "All")

# Adding the Essential Boundary Conditions
# Boundary Conditions for the upper wall
FEM.Add_EBC("y==0.01",  "UX", 0)
FEM.Add_EBC("y==0.01",  "UY", 0)
FEM.Add_EBC("y==0.01",  "DT", 0)

# Boundary Conditions for the side wall
FEM.Add_NBC("y== 2*x -0.02",  "DT", 0)
FEM.Add_NBC("y== -2*x + 0.02",  "DT", 0)

# Boundary Condition for the bottom wall
FEM.Add_EBC("y==0.0",  "DT", 18)


FEM.Analysis()

FEM.NextStep(1.0, 1.0)

print( FEM.NewtonIteration() )
print( FEM.NewtonIteration() )

ux = FEM.NodalDof("x==0.01 and y==0", "UX")
uy = FEM.NodalDof("x==0.01 and y==0", "UY")

# calculation of thickness
Thickness = np.cos(1.10715) * abs(uy)
print('ux :',ux, 'uy :',uy, 'Thickness :', Thickness )

fig, ax = plt.subplots(1,2, figsize=(21.0, 7.5))
postplot1 = FEM.ShowMesh(ax[0], deformedmesh=True, PostName="T")
ax[0].set_xlim(-0.011, 0.011)
ax[0].set_ylim(0.0, 0.01)
cbar1 = fig.colorbar(postplot1, ax=ax[0])
cbar1.set_label('absolute temperature $ {\\theta}$ in K')
postplot2 = FEM.ShowMesh(ax[1], deformedmesh=True, PostName="SigMises")
ax[1].set_xlim(-0.011, 0.011)
ax[1].set_ylim(0.0, 0.01)
cbar2 = fig.colorbar(postplot2, ax=ax[1])
cbar2.set_label('von Mises stress $\sigma_{VM}$ in Pa')
plt.show()

