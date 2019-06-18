import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
from keras.models import load_model
save_path="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/tempsave/"

x_val = np.load(open(save_path+"x_test_resampled.npy", "rb"))
def dummy(y_true, y_pred):
    return y_pred[:,0]
copyfile(save_path+"model.h5",save_path+"model_copy.h5")
model = load_model(save_path+"model_copy.h5",
        custom_objects={
            "loss_mass": dummy,
            }
            )
p = model.predict(x_val)
plt.figure()
plt.hist(p)
plt.savefig(save_path+"mass_hist")
