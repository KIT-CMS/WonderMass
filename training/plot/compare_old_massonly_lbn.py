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


#profileplotter
def profileplotter(x,y,label,pointstyle='ro', linewidth=2,linestyle="dotted",edgecolor='black',alpha=0.5,bandcolor="red"):
    def stdfunc(x):
        return np.std(x,ddof=1)
    def lower_confidence(x):
        return np.percentile(x,16)
    def upper_confidence(x):
        return np.percentile(x,84)
    #bins=[80,115,160,225,325,425,525,650,750,850,1050,1300,1550,1700,1900,2150,2500,2900,3500]
    bins=[80,105,115,125,135,160,190,225,275,325,375,425,525,650,750,850,1050,1300,1450,1550,1700,1900,2150,2450,2750,3050,3500]
    means, bin_edges, binnumber=stats.binned_statistic(x=x, values=y, statistic='mean', bins=bins, range=None)
    stds=stats.binned_statistic(x=x, values=y, statistic=stdfunc, bins=bins, range=None).statistic
    #print(stds)
    #print(means)
    lower=stats.binned_statistic(x=x, values=y, statistic=lower_confidence, bins=bins, range=None).statistic
    upper=stats.binned_statistic(x=x, values=y, statistic=upper_confidence, bins=bins, range=None).statistic
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    plt.plot(bin_centers,means,pointstyle,label=label)
    #plt.fill_between(bin_centers, means-stds, means+stds,alpha=alpha,linewidth=linewidth,linestyle=linestyle,edgecolor=edgecolor,facecolor=bandcolor)
    plt.fill_between(bin_centers, lower, upper,alpha=alpha,linewidth=linewidth,linestyle=linestyle,edgecolor=edgecolor,facecolor=bandcolor,label="1$\sigma$ confidence")
    #print("means=",means)
    #print("stds",stds)


diff_old=y_pred_old-y_true_old
diff_new=y_pred_new-y_true_new
diff_lbn=y_pred_lbn-y_true_lbn
diff_lbn2=y_pred_lbn2-y_true_lbn2
#diff_m_H=m_H-m_H_true
diff_SV=m_SVfit_pred-m_SVfit_true


plt.figure(figsize=(8,4))
profileplotter(y_true_old,diff_old/y_true_old,"network A mean",alpha=0.5,linestyle="-",edgecolor="red",bandcolor="red",pointstyle="ro")
#profileplotter(y_true_new,diff_new/y_true_new,"network B mean",alpha=0.5,linestyle="-",edgecolor="green",bandcolor="green",pointstyle="go")
#profileplotter(y_true_lbn,diff_lbn/y_true_lbn,"network C mean",alpha=0.5,linestyle="-",edgecolor="blue",bandcolor="blue",pointstyle="bo")
#profileplotter(y_true_lbn2,diff_lbn2/y_true_lbn2,"network D mean",alpha=0.5,linestyle="-",edgecolor="m",bandcolor="m",pointstyle="mo")
#profileplotter(m_SVfit_true,diff_SV/m_SVfit_true,"SVfit mean",alpha=0.5,linestyle="-",edgecolor="y",bandcolor="y",pointstyle="yo")
#profileplotter(m_H_true,diff_m_H/m_H_true,"visible mass",alpha=0.5,linestyle="-",edgecolor="green",bandcolor="green",pointstyle="go")
plt.legend()
plt.xlabel("$m_{gen}^H$ [GeV]")
plt.ylabel("$(m_{pred}^H-m_{gen}^H)/m_{gen}^H$")
plt.ylim(-0.5,0.5)
plt.tight_layout()
#plt.savefig("profil",dpi=250)

#print test loss
print("test losses")
print("network A {:.4f}".format(np.mean((diff_old/y_true_old)**2)))
print("network B {:.4f}".format(np.mean((diff_new/y_true_new)**2)))
print("network C {:.4f}".format(np.mean((diff_lbn/y_true_lbn)**2)))
print("network D {:.4f}".format(np.mean((diff_lbn2/y_true_lbn2)**2)))
print("SVfit {:.4f}".format(np.mean((diff_SV/m_SVfit_true)**2)))


#comparison histogramm (relative mass difference)
plt.figure(figsize=(8,4))
plt.hist(diff_old/y_true_old,histtype="step", lw=1, density=True, range=(-1,0.8),bins=200,label="network A",color="red")
plt.hist(diff_new/y_true_new,histtype="step", lw=1, density=True, range=(-1,0.8),bins=200,label="network B",color="green")
plt.hist(diff_lbn/y_true_lbn,histtype="step", lw=1, density=True, range=(-1,0.8),bins=200,label="network C",color="blue")
plt.hist(diff_lbn2/y_true_lbn2,histtype="step", lw=1, density=True, range=(-1,0.8),bins=200,label="network D",color="m")
plt.hist(diff_SV/m_SVfit_true,histtype="step", lw=1, density=True, range=(-1,0.8),bins=200,label="SVfit",color="y")
#plt.hist(diff_m_H/m_H_true,histtype="step", lw=1, density=True, range=(-1,0.8),bins=200,label="visible mass",color="c")
plt.xlabel("$(m_{pred}^H-m_{gen}^H)/m_{gen}^H$")
plt.legend()
plt.xlim(-0.8,0.8)
plt.tight_layout()
plt.savefig("hist")
"""
#2dhist of generated and predicted mass
def correlationplot(x,y,savename):
    plt.figure(figsize=(8,8))
    plt.hist2d(x, y,bins=[80,105,115,125,135,160,190,225,275,325,375,425,525,650,750,850,1050,1300,1450,1550,1700,1900,2150,2450,2750,3050,3500],cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("$m_{gen}^H$ [GeV]")
    plt.ylabel("$m_{pred}^H$ [GeV]")
    plt.savefig(savename,dpi=250)
    print("correlation mass generated|mass predicted")
    print(np.corrcoef(x, y=y))
correlationplot(m_SVfit_true,m_SVfit_pred,"2dhist_SVfit")
"""

#compare mass distributions and chi_square
def bin_mass_pred_to_mass_true(mass_true,mass_pred,bin_number):
    bins=[80,105,115,125,135,160,190,225,275,325,375,425,525,650,750,850,1050,1300,1450,1550,1700,1900,2150,2450,2750,3050,3500]
    binindex=np.digitize(mass_true,bins)
    #from IPython import embed; embed()
    tup=np.stack((mass_true,mass_pred,binindex),axis=-1)
    masses_pred=[]
    masses_true=[]
    for i in tup:
        if i[2]==bin_number:
            masses_true.append(i[0])
            masses_pred.append(i[1])
    return masses_true,masses_pred


def chi_square(true,pred):
    mean_true=np.mean(true)
    mean_pred=np.mean(pred)
    std_true=np.std(true,ddof=1)
    std_pred=np.std(pred,ddof=1)
    a,b=np.percentile(true,(0.5,99.5))
    bins=np.linspace(a,b,30)
    true,_=np.histogram(true,bins=bins)
    pred,_=np.histogram(pred,bins=bins)
    chi_square=sum((pred-true)**2/true)
    dof=len(true)-1

    return chi_square,dof,mean_true,mean_pred,std_true,std_pred

x1,y1=y_true_old,y_pred_old
x2,y2=y_true_new,y_pred_new
x3,y3=y_true_lbn,y_pred_lbn
x4,y4=y_true_lbn2,y_pred_lbn2
x5,y5=m_SVfit_true,m_SVfit_pred

plt.figure(figsize=(8,4))
plt.subplot(1,3,1)
true,pred=bin_mass_pred_to_mass_true(x1,y1,5)
true,pred2=bin_mass_pred_to_mass_true(x2,y2,5)
true,pred3=bin_mass_pred_to_mass_true(x3,y3,5)
true,pred4=bin_mass_pred_to_mass_true(x4,y4,5)
true5,pred5=bin_mass_pred_to_mass_true(x5,y5,5)
a,b=np.percentile(pred,(1,99))
bins=np.linspace(a,b,100)
print(chi_square(true,pred))
plt.hist(true,label="$m_{gen}^H$",bins=bins,density=True)
plt.hist(pred,label="$m_{pred}^H$ A",bins=bins,alpha=0.9,histtype="step",density=True,color="r")
plt.hist(pred2,label="$m_{pred}^H$ B",bins=bins,alpha=0.9,histtype="step",density=True,color="g")
plt.hist(pred3,label="$m_{pred}^H$ C",bins=bins,alpha=0.9,histtype="step",density=True,color="b")
plt.hist(pred4,label="$m_{pred}^H$ D",bins=bins,alpha=0.9,histtype="step",density=True,color="m")
plt.hist(pred5,label="$m_{pred}^H$ SVfit",bins=bins,alpha=0.9,histtype="step",density=True,color="y")
plt.xlabel("$m^H$ [GeV]")
plt.ylabel("Events")
#plt.legend()

plt.subplot(1,3,2)
true,pred=bin_mass_pred_to_mass_true(x1,y1,16)
true,pred2=bin_mass_pred_to_mass_true(x2,y2,16)
true,pred3=bin_mass_pred_to_mass_true(x3,y3,16)
true,pred4=bin_mass_pred_to_mass_true(x4,y4,16)
true5,pred5=bin_mass_pred_to_mass_true(x5,y5,16)
a,b=np.percentile(pred,(1,99))
bins=np.linspace(a,b,100)
print(chi_square(true,pred))
plt.hist(true,label="$m_{gen}^H$",bins=bins,density=True)
plt.hist(pred,label="$m_{pred}^H$ A",bins=bins,alpha=0.9,histtype="step",density=True,color="r")
plt.hist(pred2,label="$m_{pred}^H$ B",bins=bins,alpha=0.9,histtype="step",density=True,color="g")
plt.hist(pred3,label="$m_{pred}^H$ C",bins=bins,alpha=0.9,histtype="step",density=True,color="b")
plt.hist(pred4,label="$m_{pred}^H$ D",bins=bins,alpha=0.9,histtype="step",density=True,color="m")
plt.hist(pred5,label="$m_{pred}^H$ SVfit",bins=bins,alpha=0.9,histtype="step",density=True,color="y")
plt.xlabel("$m^H$ [GeV]")
plt.ylabel("Events")
#plt.legend()

plt.subplot(1,3,3)
true,pred=bin_mass_pred_to_mass_true(x1,y1,25)
true,pred2=bin_mass_pred_to_mass_true(x2,y2,25)
true,pred3=bin_mass_pred_to_mass_true(x3,y3,25)
true,pred4=bin_mass_pred_to_mass_true(x4,y4,25)
true5,pred5=bin_mass_pred_to_mass_true(x5,y5,25)
a,b=np.percentile(pred,(1,99))
bins=np.linspace(a,b,100)
print(chi_square(true,pred))
plt.hist(true,label="$m_{gen}^H$",bins=bins,density=True)
plt.hist(pred,label="$m_{pred}^H$ A",bins=bins,alpha=0.9,histtype="step",density=True,color="r")
plt.hist(pred2,label="$m_{pred}^H$ B",bins=bins,alpha=0.9,histtype="step",density=True,color="g")
plt.hist(pred3,label="$m_{pred}^H$ C",bins=bins,alpha=0.9,histtype="step",density=True,color="b")
plt.hist(pred4,label="$m_{pred}^H$ D",bins=bins,alpha=0.9,histtype="step",density=True,color="m")
plt.hist(pred5,label="$m_{pred}^H$ SVfit",bins=bins,alpha=0.9,histtype="step",density=True,color="y")
plt.xlabel("$m^H$ [GeV]")
plt.ylabel("Events")
plt.legend()

plt.tight_layout()
#plt.savefig("distribution",dpi=500)
