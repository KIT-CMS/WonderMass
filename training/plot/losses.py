import matplotlib.pyplot as plt
import pickle
import numpy as np

save_path1="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/best_models/final_old_1_8/"
save_path2="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/best_models/final_new_2layers/"
save_path_lbn="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/best_models/final_lbn_2_layers_1_8/"
save_path_lbn_jets="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/best_models/final_lbn_4_jets_2_layers_1_8/"
#save_path_lbn8="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/savemodeldir/"

history1=pickle.load(open(save_path1+"history.pickle", "rb"))
# history2=pickle.load(open(save_path2+"history.pickle", "rb"))
# loss_lbn_train=np.load(open(save_path_lbn+"train_loss", "rb"))
# loss_lbn_val=np.load(open(save_path_lbn+"test_loss", "rb"))
# loss_lbn_train=np.load(open(save_path_lbn+"train_loss", "rb"))
# loss_lbn_jets_val=np.load(open(save_path_lbn_jets+"test_loss", "rb"))
# loss_lbn_jets_train=np.load(open(save_path_lbn_jets+"train_loss", "rb"))
#loss_lbn_val_8=np.load(open(save_path_lbn8+"test_loss", "rb"))

a=0
plt.figure(figsize=(8,4))
plt.plot(np.array(history1["val_loss"][a:])*0.01, lw=1, label="validation loss")
plt.plot(np.array(history1["loss"][a:])*0.01, lw=1, label="training loss")
#from IPython import embed; embed()
#plt.plot(history2["val_loss_mass"][a:], lw=1, label="validation loss")
#plt.plot(history2["loss_mass"][a:], lw=1, label="training loss")
#plt.plot(loss_lbn_val[a:], lw=1, label="validation loss")
#plt.plot(loss_lbn_train[a:], lw=1, label="training loss")
#plt.plot(loss_lbn_jets_val[a:], lw=1, label="validation loss")
#plt.plot(loss_lbn_jets_train[a:], lw=1, label="training loss")
plt.legend()
plt.ylim(2.5,4)
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.savefig("1A",dpi=250)
a=0
