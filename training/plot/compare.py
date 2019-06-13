import numpy as np
np.random.seed(1234)
import pickle
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import uproot
import os
import sys
from shutil import copyfile

from keras.models import load_model

import keras.backend as K
import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.tensorflow_backend.set_session(sess)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)


#basepath = "/portal/ekpbms1/home/akhmet/merged_files_from_naf/Signal_HTT_2017_forStefan"
basepath = "/portal/ekpbms1/home/akhmet/merged_files_from_naf/DYJetToLL_2017_forStefan_newSkim"
#basepath = "/ceph/swozniewski/SM_Htautau/ntuples/Artus17_Jan/merged_fixedjets"

samples = {
    "ggh": "GluGluHToTauTauM125_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_powheg-pythia8_ext1-v1",
    "vbf": "VBFHToTauTauM125_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_powheg-pythia8_v1",
    "dy": "DYJetsToLLM50_RunIIFall17MiniAODv2_PU2017RECOSIMstep_13TeV_MINIAOD_madgraph-pythia8_v1",
    "susy100": "SUSYGluGluToHToTauTauM100_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1",
    "susy110": "SUSYGluGluToHToTauTauM110_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1",
    "susy120": "SUSYGluGluToHToTauTauM120_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1",
    "susy130": "SUSYGluGluToHToTauTauM130_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1",
    "susy140": "SUSYGluGluToHToTauTauM140_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1",
    "susy180": "SUSYGluGluToHToTauTauM180_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1",
}


def main(args):
    if not args in samples:
        raise Exception("Unknown sample.")
    sample = samples[args]

    # Load model
    def dummy(y_true, y_pred):
        return y_pred[:,0]
    copyfile("model.h5", "model_copy.h5")
    model = load_model("model_copy.h5",
            custom_objects={
                "loss_p3_1": dummy,
                "loss_p3_2": dummy,
                "loss_p3_h": dummy,
                "loss_f_1": dummy,
                "loss_f_2": dummy,
                "loss_p3_h": dummy,
                "loss_p_1": dummy,
                "loss_p_2": dummy,
                "loss_p_h": dummy,
                "loss_mass": dummy,
                "loss_pt": dummy,
                "loss_eta": dummy,
                "loss_phi": dummy,
                "loss_p": dummy,
                "loss": dummy,
                            })

    # Load data
    branches = ["pt_1", "eta_1", "phi_1", "m_1", "pt_2", "eta_2", "phi_2", "m_2", "met", "metphi",
                "m_sv", "pt_sv", "eta_sv", "phi_sv",
                "genbosonpt", "genbosoneta", "genbosonphi", "genbosonmass",
                "gen_match_1", "gen_match_2"]
    file_ = uproot.open(os.path.join(basepath, sample, sample + ".root"))
    tree = file_["tt_nominal"]["ntuple"].arrays(branches)

    data = {}
    for b in branches:
        data[b] = tree[b.encode('ascii')]

    """
    mask = (data["gen_match_1"] == 5) * (data["gen_match_2"] == 5)
    for b in branches:
        data[b] = data[b][mask]
    """

    for i in ["1", "2"]:
        data["t%s_rec_px"%i] = data["pt_" + i] * np.cos(data["phi_" + i])
        data["t%s_rec_py"%i] = data["pt_" + i] * np.sin(data["phi_" + i])
        data["t%s_rec_pz"%i] = data["pt_" + i] * np.sinh(data["eta_" + i])
        data["t%s_rec_e"%i] = np.sqrt(data["t%s_rec_px"%i]**2 + data["t%s_rec_py"%i]**2\
                               + data["t%s_rec_pz"%i]**2 + data["m_" + i]**2)
    data["met_rec_px"] = data["met"] * np.cos(data["metphi"])
    data["met_rec_py"] = data["met"] * np.sin(data["metphi"])

    # Plot inputs
    inputs = pickle.load(open("x.pickle", "rb"))
    outputs_t = pickle.load(open("y_t.pickle", "rb"))

    nbins = 20
    """
    for key in inputs:
        plt.figure(figsize=(6,6))
        q = np.percentile(data[key], [1, 99])
        plt.hist(data[key], histtype="step", lw=3, range=q, bins=nbins)
        plt.xlabel(key)
        plt.xlim(q)
        plt.savefig(key + "_" + args + "_c.png")
    """

    # Make prediction
    x = np.vstack([data[key] for key in inputs]).T
    p = model.predict(x)
    pred = {}
    for i, k in enumerate(outputs_t):
        pred[k] = p[:,i]

    # Add energy
    tau_mass = 1.77
    for i in ["1", "2"]:
        for d in [pred]:
            d["t%s_gen_e"%i] = np.sqrt(tau_mass**2 + d["t%s_gen_px"%i]**2 + d["t%s_gen_py"%i]**2 + d["t%s_gen_pz"%i]**2)

    # Add Higgs system
    for c in ["px", "py", "pz", "e"]:
        pred["h_gen_%s"%c] = pred["t1_gen_%s"%c] + pred["t2_gen_%s"%c]

    pred["genbosonpt"] = np.sqrt(pred["h_gen_px"]**2 + pred["h_gen_py"]**2)
    mag = np.sqrt(pred["h_gen_px"]**2 + pred["h_gen_py"]**2 + pred["h_gen_pz"]**2)
    costheta = pred["h_gen_pz"] / mag
    pred["genbosoneta"] = -0.5 * np.log((1.0 - costheta) / (1.0 + costheta))
    pred["genbosonphi"] = np.arctan2(pred["h_gen_py"], pred["h_gen_px"])
    pred["genbosonmass"] = np.sqrt(np.abs(pred["h_gen_e"]**2 - pred["h_gen_px"]**2\
                                  - pred["h_gen_py"]**2 - pred["h_gen_pz"]**2))

    # Print
    def printMeanStd(d, label):
        print("{:}: {:.2f} +- {:.2f}".format(label, np.mean(d), np.std(d)))

    def printMedian(d, label):
        p = np.percentile(d, (50))
        print("{:}: median: {:.2f}".format(label, p))

    def printIntervals(d, intervals, label):
        for ps in intervals:
            p = np.percentile(d, (100 - ps, ps))
            print("{:}: interval ({:}): ({:.2f}, {:.2f})".format(label, ps, p[0], p[1]))

    def printAll(d, label):
        printMeanStd(d, label)
        printMedian(d, label)
        printIntervals(d, [68, 95, 99], label)

    print("Sample: " + args)
    printAll(data["m_sv"], "SVFIT")
    printAll(pred["genbosonmass"], "NN")

    # Plot Higgs properties
    ranges = {"mass": (50, 200), "pt": (0, 100), "phi": (-np.pi, np.pi), "eta": (-4.0, 4.0)}
    for label, svfit, nn in [
            ["mass", "m_sv", "genbosonmass"],
            ["pt", "pt_sv", "genbosonpt"],
            ["eta", "eta_sv", "genbosoneta"],
            ["phi", "phi_sv", "genbosonphi"]]:
        # Marginal distribution
        plt.figure(figsize=(6,6))
        bins = nbins
        p = ranges[label]
        c1, _, _ = plt.hist(data[svfit], histtype="step", lw=3, range=p, bins=bins, alpha=0.8, label="SVFIT")
        c2, _, _ = plt.hist(pred[nn], histtype="step", lw=3, range=p, bins=bins, alpha=0.8, label="NN")
        plt.hist(data[nn], histtype="step", lw=3, range=p, bins=bins, alpha=0.8, label="truth")
        plt.legend()
        plt.xlim(p)
        plt.ylim((0, max(max(c1), max(c2)) * 1.3))
        plt.xlabel(label)
        plt.savefig(label + "_" + args + ".png")

        # Correlation
        plt.figure(figsize=(6,6))
        plt.hist2d(data[svfit], pred[nn],
                range=(ranges[label], ranges[label]),
                bins=(100, 100),
                norm=mpl.colors.LogNorm())
        plt.xlim(ranges[label])
        plt.ylim(ranges[label])
        plt.xlabel("SVFIT")
        plt.ylabel("NN")
        plt.savefig(label + "_corr_" + args + ".png")

    # Save response
    r = {}
    r["m_sv"] = data["m_sv"]
    r["m_h"] = pred["genbosonmass"]
    pickle.dump(r, open(sample[0:3] + ".pickle", "wb"))


if __name__ == "__main__":
    args = sys.argv[1]
    main(args)
