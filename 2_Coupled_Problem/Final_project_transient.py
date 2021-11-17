# ----------------------------------------------------------------------------------------------------------------------
#
# Description: The following code solves the boundary value problem, thermo mechanical transient analysis of steel 
#              clading
#
# Author:      V Mithlesh Kumar, vmithleshkumar@gmail.com
#
# Reference:  Sascha Maassen, Finite Element Method - Coupled Problems code
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import PyFEMP
import Q1_LE_Dyn_T_template as ELEMENT

FEM = PyFEMP.FEM_Simulation(ELEMENT)
n = 8                                                    
XI, Elem = PyFEMP.msh_conv_quad([-0.01, 0.0], [-0.005, 0.01], [0.005, 0.01], [0.01, 0.0], [3*n, n], type='Q1')
FEM.Add_Mesh(XI, Elem)
# "E", "nu", "rho", "eta", "bx", "by", "a_q", "c", "r", "T0", "a_T"
FEM.Add_Material([210e09, 0.3, 7850, 0.0, 0.0, 0.0, 50.0, 460.0, 0.0, 290.15, 7e-6], "All")

# Adding the Boundary Conditions
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

time  = 0.0
dt    = 4096
nStep = 22

animation = True
times = [0]
uys   = [0]
uxs   = [0]

def load(time):

    Lambda = np.sin(np.pi*time/(12*60*60))

    return Lambda

for step in range(nStep):
  time += dt
  FEM.NextStep(time, load(time))
  print( FEM.NewtonIteration() )
  print( FEM.NewtonIteration() )

  ux = FEM.NodalDof("x==0.01 and y==0", "UX")
  uy = FEM.NodalDof("x==0.01 and y==0", "UY")
  

  print('ux :', ux, 'uy :', uy)

  times.append(time)
  uys.append(uy)
  uxs.append(ux)
  
  if animation:
    if (step==0): fig, ax = plt.subplots(1,1, figsize=(40.0, 4.0))
    ax.cla()
    postplot1 = FEM.ShowMesh(ax, deformedmesh=True, PostName="T")
    plt.pause(0.00001)
  

plt.show()


# save data in files for later post processing
np.save('time_data',times)
np.save('x_disp_data',uxs)
np.save('y_disp_data',uys)




