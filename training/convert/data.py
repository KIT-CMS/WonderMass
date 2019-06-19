import uproot
import numpy as np
np.random.seed(1234)
import pickle
from multiprocessing import Pool
import os
from functools import partial


path = "output.root"
save_path="tempsave/"

def resample(x):
    px1 = x[:, 0]
    py1 = x[:, 1]
    pz1 = x[:, 2]
    px2 = x[:, 3]
    py2 = x[:, 4]
    pz2 = x[:, 5]
    m_t = 1.77
    e1 = np.sqrt(m_t**2 + px1**2 + py1**2 + pz1**2)
    e2 = np.sqrt(m_t**2 + px2**2 + py2**2 + pz2**2)
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2
    m = np.sqrt(e**2 - px**2 - py**2 - pz**2)

    p = np.percentile(m, [0.5, 99.5])
    counts, bins = np.histogram(m, bins=20, range=p)
    print("Min/max resampling mass range: {}/{}".format(p[0], p[-1]))

    hist_idx = np.digitize(
            np.clip(m, bins[0], bins[-1] - (bins[-1] - bins[-2]) * 0.5),
            bins)
    times = np.array(np.round(np.max(counts) / counts[hist_idx - 1]), dtype=int)
    print("Min/max resampling factor: {}/{}".format(min(times), max(times)))

    from sklearn.model_selection import train_test_split
    idx = range(x.shape[0])
    times_train, times_test, idx_train, idx_test, = train_test_split(times, idx, train_size=0.8, random_state=1234)

    idx_train_re = []
    idx_test_re = []
    for i, j in zip(idx_train, times_train):
        idx_train_re += [i] * j
    for i, j in zip(idx_test, times_test):
        idx_test_re += [i] * j

    idx_re = np.array(idx_train_re + idx_test_re)
    idx_train_re = np.array(idx_train_re)
    idx_test_re = np.array(idx_test_re)

    counts_after, _ = np.histogram(m[idx_re], bins=bins)
    pickle.dump([counts, bins, counts_after], open(save_path+"resample.pickle", "wb"))

    return idx_re, idx_train_re, idx_test_re


def resample_new(x):
    px1 = x[:, 0]
    py1 = x[:, 1]
    pz1 = x[:, 2]
    px2 = x[:, 3]
    py2 = x[:, 4]
    pz2 = x[:, 5]
    m_t = 1.77
    e1 = np.sqrt(m_t**2 + px1**2 + py1**2 + pz1**2)
    e2 = np.sqrt(m_t**2 + px2**2 + py2**2 + pz2**2)
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2
    m = np.sqrt(e**2 - px**2 - py**2 - pz**2)

    #only events within this tow percentiles are used
    p = np.percentile(m, [0.5, 99.5])

    """
    counts, bins = np.histogram(m, bins=20, range=p)
    plt.figure()
    plt.plot(bins[0:20]+(bins[1]-bins[0])/2,counts,".")
    plt.savefig("testhist.png")
    """
    bins=np.linspace(p[0],p[1],num=41)
    #print(bins)
    indices=[]
    masses=[]
    for i in range(len(bins)-1):
        indexlist=[]
        #masslist=[]
        for n,mass in enumerate(m):
            if bins[i]<=mass and mass<bins[i+1]:
                indexlist.append(n)
                #masslist.append(mass)
        indices.append(indexlist)
        #masses.append(masslist)

    count=0
    for indexlist in indices:
        if len(indexlist)>count:
            count=len(indexlist)
    print(count)
    indices_res=[]
    for indexlist in indices:
        a=indexlist
        while len(a)<count-1000:
            a+=list(np.random.choice(indexlist,1000))
        while len(a)<count-100:
            a+=list(np.random.choice(indexlist,100))
        while len(a)<count:
            a+=list(np.random.choice(indexlist,1))
        indices_res+=a

    print("Min/max resampling mass range: {}/{}".format(p[0], p[-1]))
    from sklearn.model_selection import train_test_split
    idx_re = np.array(indices_res)
    idx_train_re,idx_test_re = train_test_split(idx_re, train_size=0.8, random_state=1234)
    idx_train_re=np.array(idx_train_re)
    idx_test_re=np.array(idx_test_re)
    """
    plt.figure()
    plt.hist(m[idx_train_re],bins=20)
    plt.savefig("testhistafter.png")


    counts_after, _ = np.histogram(m[idx_re], bins=bins)
    pickle.dump([counts, bins, counts_after], open(save_path+"resample.pickle", "wb"))
    """
    return idx_re, idx_train_re, idx_test_re, m


def main():
    # Declare inputs and outputs
    components = ["px", "py", "pz", "e"]
    components_3 = ["px", "py", "pz"]
    inputs = ["t1_rec_" + c for c in components]\
           + ["t2_rec_" + c for c in components]\
           + ["met_rec_" + c for c in ["px", "py"]]
    outputs_n = ["met_gen_" + c for c in components]
    outputs_t = ["t1_gen_" + c for c in components_3]\
              + ["t2_gen_" + c for c in components_3]
    outputs_h = ["h_gen_" + c for c in components]
    pickle.dump(inputs, open(save_path+"x.pickle", "wb"))
    pickle.dump(outputs_n, open(save_path+"y_n.pickle", "wb"))
    pickle.dump(outputs_t, open(save_path+"y_t.pickle", "wb"))
    pickle.dump(outputs_h, open(save_path+"y_h.pickle", "wb"))

    # Load data
    data = {}
    tree = uproot.open(path)["ntupleBuilder"]["Events"].arrays()
    for b in tree:
        data[b.decode("utf-8")] = tree[b]

    # Stack arrays
    x = np.vstack([data[key] for key in inputs]).T
    y_n = np.vstack([data[key] for key in outputs_n]).T
    y_t = np.vstack([data[key] for key in outputs_t]).T
    y_h = np.vstack([data[key] for key in outputs_h]).T

    # Resample events to flat mass
    idx, idx_train, idx_test, m = resample_new(y_t)
    x_resampled = x[idx]
    y_t_resampled = y_t[idx]
    print("x: Before/after resampling: {} / {}".format(x.shape, x_resampled.shape))
    print("y_t: Before/after resampling: {} / {}".format(y_t.shape, y_t_resampled.shape))

    # Save to disk
    np.save(open(save_path+"x.npy", "wb"), x)
    np.save(open(save_path+"x_resampled.npy", "wb"), x_resampled)
    np.save(open(save_path+"x_train_resampled.npy", "wb"), x[idx_train])
    np.save(open(save_path+"x_test_resampled.npy", "wb"), x[idx_test])
    np.save(open(save_path+"y_n.npy", "wb"), y_n)
    np.save(open(save_path+"y_t.npy", "wb"), y_t)
    np.save(open(save_path+"y_t_resampled.npy", "wb"), y_t_resampled)
    np.save(open(save_path+"y_t_train_resampled.npy", "wb"), y_t[idx_train])
    np.save(open(save_path+"y_t_test_resampled.npy", "wb"), y_t[idx_test])
    np.save(open(save_path+"y_h.npy", "wb"), y_h)
    np.save(open(save_path+"mass_train_npy","wb"), m[idx_train])
    np.save(open(save_path+"mass_test_npy","wb"), m[idx_test])



if __name__ == "__main__":
    main()
