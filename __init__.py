from pyprocessmacro import Process
import numpy as np
import pandas as pd

df = pd.DataFrame({"Y": np.random.normal(0, 1, 1000),
                   "X": np.random.normal(0, 1, 1000),
                   "M1": np.random.choice([0, 1], 1000),
                   "M2": np.random.choice([0, 1], 1000),
                   "Med": np.random.normal(0, 1, 1000)})
p = Process(data=df, x="X", y="Y", model=10, z="M1", w="M2", m="Med")
dfi = p.get_conditional_indirect_effects("Med", {"M1":[1, 2, 3]})
print(dfi)
dfd = p.get_conditional_direct_effects({"M2":[1, 2, 3]})
print(dfd)
p.plot_direct_effects(x="M2", mods_at={"M2":[1, 2, 3]})



p = Process(data=df, x="X", y="Y", model=4, m="Med")
print(p.summary())
dfi = p.get_conditional_indirect_effects("Med")
print(dfi)
dfd = p.get_conditional_direct_effects()
print(dfd)
