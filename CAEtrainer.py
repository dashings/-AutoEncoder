import numpy as np
import matplotlib.pyplot as plt
import ConvAutoEncoder
from ConvAutoEncoder import ConvAutoEncoder as CAE
model=CAE()
model.trainNN(lr=0.003, weight_decay=1e-5, epochs=18)

x,y=model.predict(ConvAutoEncoder.trainData[0])
plt.imshow(x)
plt.show()
plt.imshow(y)
plt.show()
model.show()
