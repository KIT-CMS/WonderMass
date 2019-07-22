import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import uproot


save_path1="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/saved_models/saves_final_old/"
save_path2="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/tempsave/"
save_path3="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/saved_models/savemodeldir/"
save_path4="/local/scratch/ssd2/hborsch/di_tau_mass/DNN/Wondermass/saved_models/saves_lbn_18_07/"

"""
for j,save_path in enumerate([save_path1,save_path2]):
    if j==0:
        word="old NN"
    elif j==1:
        word="new NN"
    history = pickle.load(open(save_path+"history.pickle", "rb"))
    loss_train=history["loss_mass"]
    loss_val=history["val_loss_mass"]
    a=15
    epochs = range(1, len(history["loss"])+1)
    plt.plot(epochs[a:], loss_val[a:], lw=1, label="Validation loss "+word)
    #plt.plot(epochs[a:], loss_train[a:], lw=1, label="Training loss"+word)
    print(loss_val[-1],word)
    plt.legend()
    plt.ylim(0,60)
    plt.xlabel("Epochs")
    plt.ylabel("mass_loss")
    plt.savefig("loss_curve_val_train_2")
"""

#compare performance of all networks
y_true_lbn=np.squeeze(np.load(open(save_path3+"y_test","rb")))
y_pred_lbn=np.load(open(save_path3+"y_test_pred","rb"))
y_true_lbn2=np.squeeze(np.load(open(save_path4+"y_test","rb")))
y_pred_lbn2=np.load(open(save_path4+"y_test_pred","rb"))


y_true_lbn=y_true_lbn[:-y_true_lbn.shape[0] + y_pred_lbn.shape[0]]
y_true_lbn2=y_true_lbn2[:-y_true_lbn2.shape[0] + y_pred_lbn2.shape[0]]
y_true_old,y_pred_old=np.load(open(save_path1+"mass_test_true","rb")),np.load(open(save_path1+"mass_test_pred","rb"))

y_true_new,y_pred_new=np.load(open(save_path2+"mass_test_npy","rb")),np.load(open(save_path2+"mass_test_pred","rb"))


"""
#read in visible mass
components = ["e","px", "py", "pz"]
inputs_t1=["t1_rec_" + c for c in components]
inputs_t2=["t2_rec_" + c for c in components]
data = {}
tree = uproot.open("MSSM_merged.root")["ntupleBuilder"]["Events_tt"].arrays()

for b in tree:
    data[b.decode("utf-8")] = tree[b]
vec_t1=np.stack([data[name] for name in inputs_t1],axis=-1)
vec_t2=np.stack([data[name] for name in inputs_t2],axis=-1)
m_H_true=data["h_gen_mass"]
vec_H=vec_t1+vec_t2
m_H=np.sqrt(vec_H[:,0]**2-vec_H[:,1]**2-vec_H[:,2]**2-vec_H[:,3]**2)
#m_H_train,m_H_test,m_H_true_train,m_H_true_test = train_test_split(m_H,m_H_true,test_size=0.2,random_state=1234)

#reading svfit masses, true and pred
data={}
tree = uproot.open("MSSM_merged_gen_for_SVfit.root")["tt_nominal"]["ntuple"].arrays()
for b in tree:
    data[b.decode("utf-8")] = tree[b]
m_SVfit_true=data["genbosonmass"]
data={}
tree = uproot.open("MSSM_merged_SVfit_tt.root")["tt_nominal"]["ntuple"].arrays()
for b in tree:
    data[b.decode("utf-8")] = tree[b]
m_SVfit_pred=data["m_sv"]
"""

def profileplotter(x,y,label,pointstyle='ro', linewidth=2,linestyle="dotted",edgecolor='black',alpha=0.5,bandcolor="red"):
    def stdfunc(x):
        return np.std(x,ddof=1)
    bins=[80,150,225,325,425,525,650,750,850,1050,1300,1550,1700,1900,2150,2500,2900,3500]
    means, bin_edges, binnumber=stats.binned_statistic(x=x, values=y, statistic='mean', bins=bins, range=None)
    stds=stats.binned_statistic(x=x, values=y, statistic=stdfunc, bins=bins, range=None).statistic
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    plt.plot(bin_centers,means,pointstyle,label=label)
    plt.fill_between(bin_centers, means-stds, means+stds,alpha=alpha,linewidth=linewidth,linestyle=linestyle,edgecolor=edgecolor,facecolor=bandcolor)
p=[-600,500]

diff_lbn=y_pred_lbn-y_true_lbn
diff_lbn2=y_pred_lbn2-y_true_lbn2
diff_old=y_pred_old-y_true_old
diff_new=y_pred_new-y_true_new
#diff_m_H=m_H-m_H_true
#diff_SV=m_SVfit_pred-m_SVfit_true

plt.figure(figsize=(8,8))
#profileplotter(y_true_new,diff_new/y_true_new,"new",alpha=0.5,linestyle="-",edgecolor="blue",bandcolor="blue",pointstyle="bo")
profileplotter(y_true_lbn,diff_lbn2/y_true_lbn,"lbn_4jets",alpha=0.5,linestyle="-",edgecolor="red",bandcolor="red",pointstyle="ro")
profileplotter(y_true_lbn2,diff_lbn/y_true_lbn2,"lbn",alpha=0.5,linestyle="-",edgecolor="blue",bandcolor="blue",pointstyle="bo")
#profileplotter(y_true_old,diff_old/y_true_old,"old",alpha=0.5,linestyle="-",edgecolor="green",bandcolor="green",pointstyle="go")
#profileplotter(m_SVfit_true,diff_SV/m_SVfit_true,"SVfit",alpha=0.5,linestyle="-",edgecolor="green",bandcolor="green",pointstyle="go")
#profileplotter(m_H_true,diff_m_H/m_H_true,"visible mass",alpha=0.5,linestyle="-",edgecolor="green",bandcolor="green",pointstyle="go")
plt.legend()
plt.xlabel("generated mass")
plt.ylabel("(m_pred-m_gen)/mass_gen")
plt.ylim(-0.5,0.5)
plt.savefig("profil_hist_5",dpi=250)
