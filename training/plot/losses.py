import matplotlib.pyplot as plt
import pickle
import numpy as np

save_path1="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/saved_models/saves_old_MSSM_not_res/"
save_path2="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/tempsave/"
save_path_lbn="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/saved_models/lbn_saves_19_07/"


history1=pickle.load(open(save_path1+"history.pickle", "rb"))
history2=pickle.load(open(save_path2+"history.pickle", "rb"))
loss_lbn_train=np.load(open(save_path_lbn+"train_loss", "rb"))
loss_lbn_val=np.load(open(save_path_lbn+"test_loss", "rb"))

a=15
#plt.plot(history1["val_loss_mass"][a:], lw=1, label="Validation loss old")
#plt.plot(history1["loss_mass"][a:], lw=1, label="Training loss old")
plt.plot(history2["val_loss_mass"][a:], lw=1, label="Validation loss new")
#plt.plot(history2["loss_mass"][a:], lw=1, label="Training loss new")
plt.plot(loss_lbn_val[a:], lw=1, label="Validation loss lbn")
#plt.plot(loss_lbn_train[a:], lw=1, label="Training loss lbn")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("mass_loss")
plt.savefig("mass_losses")
