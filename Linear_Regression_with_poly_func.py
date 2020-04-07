import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi, 1000)
y = 100*np.sin(x) + 0.5*np.exp(x) +300
err = np.random.rand(1000)
t = y+err

phi1 = np.array([1 for i in range(1000)])[:,np.newaxis]
phi = phi1.copy()
fig, axes = plt.subplots(5, 4, sharex=True,sharey=True, figsize=(15,15))
mse_list = []
count = 1
for row in range(1,6):
    for column in range(1,5):
        phi = np.concatenate((phi,x[:,np.newaxis]**count),axis=1)
        phi2 = np.linalg.inv(np.dot(phi.T,phi)).dot(phi.T)
        w = phi2.dot(t)
        new_y = (w*phi).sum(axis=1)
        axes[row-1,column-1].plot(x,t,'b-', label='Оригинальный график')
        axes[row-1,column-1].plot(x,new_y,'y-', label='График полинома')
        axes[row-1,column-1].set_xlabel('X')
        axes[row-1,column-1].set_ylabel('Y')
        axes[row-1,column-1].legend()
        mse = ((t-new_y)**2).sum()/1000
        axes[row-1,column-1].set_title('Power {0} \n MSE = {1}'.format(count, np.round(mse,0)))
        mse_list.append(mse)
        count += 1
plt.ylim((-50,500))
plt.subplots_adjust(wspace=1, hspace=1)
plt.show()

x = np.arange(1,21)
y = np.array(mse_list)
fig, ax = plt.subplots()
ax.plot(x,y)
plt.ylim((0,10000000))
ax.set_xlabel('Max power')
ax.set_ylabel('MSE')
plt.show()
