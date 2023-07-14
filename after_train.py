import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

alllosses = np.loadtxt("./DataLog/alllosses.csv")
vallosses = np.loadtxt("./DataLog/vallosses.csv")

all_train_mape = np.loadtxt("./DataLog/all_train_mape.csv")
all_train_rmse = np.loadtxt("./DataLog/all_train_rmse.csv")
all_val_mape = np.loadtxt("./DataLog/all_val_mape.csv")
all_val_rmse = np.loadtxt("./DataLog/all_val_rmse.csv")


plt.plot(alllosses)
plt.title("loss on train_set during training processing")
plt.savefig("./figout/train_loss.png")
plt.show()

plt.plot(vallosses)
plt.title("loss on valid_set during training processing")
plt.savefig("./figout/valid_loss.png")
plt.show()

plt.plot(all_train_mape)
plt.title("mape on train_set during training processing")
plt.savefig("./figout/train_mape.png")
plt.show()

plt.plot(all_val_mape)
plt.title("mape on valid_set during training processing")
plt.savefig("./figout/valid_mape.png")
plt.show()

plt.plot(all_train_rmse)
plt.title("rmse on train_set during training processing")
plt.savefig("./figout/train_rmse.png")
plt.show()

plt.plot(all_val_rmse)
plt.title("rmse on valid_set during training processing")
plt.savefig("./figout/valid_rmse.png")
plt.show()
