import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
from keras.models import load_model
import pickle

save_path="tempsave/"
#save_path="saves_2_MSSM_not_resampled_absolute_loss/"


x_val = np.load(open(save_path+"x_test_resampled.npy", "rb"))
mass_val=np.load(open(save_path+"mass_test_npy","rb"))
def dummy(y_true, y_pred):
    return y_pred[:,0]
copyfile(save_path+"model.h5",save_path+"model_copy.h5")
model = load_model(save_path+"model_copy.h5",
        custom_objects={
            "loss_mass": dummy,
            }
            )

p = np.array(model.predict(x_val))
p=np.squeeze(p)

print(p.shape,mass_val.shape)

plt.figure()
plt.hist((p-mass_val)/mass_val,
        histtype="step", lw=3, alpha=0.8, bins=np.linspace(-1.5,1.5,100))
plt.xlim(-1,1)
plt.ylim(0,22000)
plt.savefig(save_path+"mass_rel_hist")

plt.figure()
plt.hist(p)
plt.savefig(save_path+"mass_hist_pred")

plt.figure(figsize=(10,10),dpi=1000)
plt.hist(mass_val,bins=1500)
plt.savefig(save_path+"mass_hist_true")
plt.figure(figsize=(4,4))
diff=mass_val-p
p = np.percentile(diff, [1, 99])
_, bins, _ = plt.hist(diff,
        histtype="step", lw=3, alpha=0.8, range=p, bins=30)
plt.savefig(save_path+"mass_diff_hist_98")


history = pickle.load(open(save_path+"history.pickle", "rb"))
a=5
epochs = range(1, len(history["loss"])+1)
for key in history.keys():
    if key=="loss":
        plt.figure()
        #plt.plot(epochs[a:], history[key][a:], lw=1, label="Training loss ")
        plt.plot(epochs[a:], history["val_" + key][a:], lw=1, label="Validation loss ")
        print(min(history["val_" + key][a:]))
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(key)
        plt.savefig(save_path+"loss")
