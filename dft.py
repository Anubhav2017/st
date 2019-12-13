import numpy as np
import matplotlib.pyplot as plt

x=[]
for i in range(100):
	x.append(np.random.randint(0,1024))

x=np.array(x)
x=np.sort(x)
y=np.fft.fft(x)
xxs=np.fft.fftfreq(x.shape[-1])
print(xxs)

plt.plot(xxs,x,xxs,y)
plt.show()