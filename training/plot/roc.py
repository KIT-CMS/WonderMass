import numpy as np
np.random.seed(1234)
import pickle
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys


def main(args):
    bkg_tag = "DYJ"
    if "vbf" in args:
        sig_tag = "VBF"
    elif "ggh" in args:
        sig_tag = "Glu"
    else:
        raise Exception("Unknown tag.")
    bkg_sv = pickle.load(open(bkg_tag + ".pickle", "rb"))["m_sv"]
    bkg_nn = pickle.load(open(bkg_tag + ".pickle", "rb"))["m_h"]
    sig_sv = pickle.load(open(sig_tag + ".pickle", "rb"))["m_sv"]
    sig_nn = pickle.load(open(sig_tag + ".pickle", "rb"))["m_h"]

    true_sv = np.hstack([np.zeros(bkg_sv.size), np.ones(sig_sv.size)])
    pred_sv = np.hstack([bkg_sv, sig_sv])
    fpr_sv, tpr_sv, t_sv = roc_curve(true_sv, pred_sv)
    auc_sv = auc(fpr_sv, tpr_sv)
    print("AUC (SVFIT): {:.4f}".format(auc_sv))

    true_nn = np.hstack([np.zeros(bkg_nn.size), np.ones(sig_nn.size)])
    pred_nn = np.hstack([bkg_nn, sig_nn])
    fpr_nn, tpr_nn, t_nn = roc_curve(true_nn, pred_nn)
    auc_nn = auc(fpr_nn, tpr_nn)
    print("AUC (NN): {:.4f}".format(auc_nn))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_sv, tpr_sv, lw=3, alpha=0.8, label="SVFIT (AUC={:.2f})".format(auc_sv))
    plt.plot(fpr_nn, tpr_nn, lw=3, alpha=0.8, label="NN (AUC={:.2f})".format(auc_nn))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig("roc_{}_{}.png".format(bkg_tag, sig_tag))


if __name__ == "__main__":
    args = sys.argv[1]
    main(args)
