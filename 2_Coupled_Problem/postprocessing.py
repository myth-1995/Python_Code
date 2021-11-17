import numpy as np
import matplotlib.pyplot as plt

# load data saved after a simulation
time = np.load('time_data_9.npy')
uxs  = np.load('x_disp_data_9.npy')
uys  = np.load('y_disp_data_9.npy')

# plot the displacement over time
fig, ax = plt.subplots(1,1, figsize=(6.0, 5.0))
ax.plot(time, uys, color='red', label="y-disp")
ax.plot(time, uxs, color='blue', label="x-disp")
ax.set_xlim(0, max(time))
ax.set_ylim(-0.00001, 0.00001)
ax.set_xlabel('time in s')
ax.set_ylabel('vertical displacement in m')
ax.legend()
fig.suptitle('A comparison of vertical and horizontal displacement in time \n for x=0.01,y=0')
plt.show()

max_x=max(uxs)
print ('maximum x value :' , max_x)
max_y=max(uys)
print ('maximum y value :' , max_y)
Thickness = np.cos(1.10715)*max_y
print('Thickness:',Thickness)
