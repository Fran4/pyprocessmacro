from pyprocessmacro import Process
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.normal(0, 1, 1000)
m1 = np.random.choice([0, 1], 1000)
m2 = np.random.normal(0, 1, 1000)

med = 0.3*x - 0.3*x*m1 - 0.2*x*m2 + 0.3*x*m1*m2 + np.random.normal(0, 0.1, 1000)
y = 0.3*x - 0.3*x*m1 - 0.2*x*m2 + 0.3*x*m1*m2 + 0.3*med + np.random.normal(0, 0.1, 1000)

df = pd.DataFrame({"Y": y,
                   "X": x,
                   "M1": m1,
                   "M2": m2,
                   "Med": med})
p = Process(data=df, x="X", y="Y", model=12, z="M1", w="M2", m="Med")
dfi = p.get_conditional_indirect_effects("Med", modval={"M2":[0, 1], "M1": [1, 2, 3]})
print(dfi)
dfd = p.get_conditional_direct_effects(modval={"M2":[0, 1], "M1":[1, 2, 3]})
print(dfd)
g = p.plot_conditional_direct_effects(x="M2", hue="M1", modval={"M2":[0, 1], "M1": [1, 2, 3]})
plt.show()
p.plot_conditional_indirect_effects("Med", x="M2", modval={"M2":[0, 1], "M1":[100]})
plt.show()
print(p.summary())