import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import uproot
import os
import numpy as np

#basepath = ""
#basepath = "/ceph/swozniewski/SM_Htautau/ntuples/Artus17_Jan/merged_fixedjets"
basepath = "/portal/ekpbms1/home/akhmet/merged_files_from_naf/Signal_HTT_2017_forStefan"
samplename = "GluGluHToTauTauM125_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_powheg-pythia8_ext1-v1"

simpath = "output_125.root"

def main():
    # Load official data
    branches = ["pt_1", "eta_1", "phi_1", "m_1", "pt_2", "eta_2", "phi_2", "m_2",
                "q_1", "q_2", "decayMode_1", "decayMode_2",
                "met", "metphi",
                "m_sv", "pt_sv", "eta_sv", "phi_sv",
                "genbosonpt", "genbosoneta", "genbosonphi", "genbosonmass"]
    file_ = uproot.open(os.path.join(basepath, samplename, samplename + ".root"))
    tree = file_["tt_nominal"]["ntuple"].arrays(branches)

    data = {}
    for b in branches:
        data[b] = tree[b.encode("ascii")]

    for i in ["1", "2"]:
        data["t%s_rec_q"%i] = data["q_" + i]
        data["t%s_rec_dm"%i] = data["decayMode_" + i]
        data["t%s_rec_px"%i] = data["pt_" + i] * np.cos(data["phi_" + i])
        data["t%s_rec_py"%i] = data["pt_" + i] * np.sin(data["phi_" + i])
        data["t%s_rec_pz"%i] = data["pt_" + i] * np.sinh(data["eta_" + i])
        data["t%s_rec_e"%i] = np.sqrt(data["t%s_rec_px"%i]**2 + data["t%s_rec_py"%i]**2\
                               + data["t%s_rec_pz"%i]**2 + data["m_" + i]**2)
    data["met_rec_px"] = data["met"] * np.cos(data["metphi"])
    data["met_rec_py"] = data["met"] * np.sin(data["metphi"])

    # Load simulation
    branches = []
    readonly_branches = []
    for t in ["1", "2"]:
        for x in ["px", "py", "pz", "e", "dm", "q"]:
            branches.append("t%s_rec_%s"%(t, x))
            if not x in ["dm", "q"]:
                readonly_branches.append("t%s_gen_%s"%(t, x))
    branches += ["met_rec_px", "met_rec_py"]

    file_ = uproot.open(simpath)
    tree = file_["ntupleBuilder"]["Events"].arrays(branches + readonly_branches)

    sim = {}
    for b in branches + readonly_branches:
        sim[b] = tree[b.encode("ascii")]

    for p in ["px", "py", "pz", "e"]:
        sim["tb_gen_%s"%p] = sim["t1_gen_%s"%p] + sim["t2_gen_%s"%p]

    for g in ["rec", "gen"]:
        types = ["1", "2"] if g == "rec" else ["1", "2", "b"]
        for t in types:
            sim["pt_%s_%s"%(g,t)] = np.sqrt(sim["t%s_%s_px"%(t,g)]**2 + sim["t%s_%s_py"%(t,g)]**2)
            mag = np.sqrt(sim["t%s_%s_px"%(t,g)]**2 + sim["t%s_%s_py"%(t,g)]**2 + sim["t%s_%s_pz"%(t,g)]**2)
            costheta = sim["t%s_%s_pz"%(t,g)] / mag
            sim["eta_%s_%s"%(g,t)] = -0.5 * np.log((1.0 - costheta) / (1.0 + costheta))
            sim["phi_%s_%s"%(g,t)] = np.arctan2(sim["t%s_%s_py"%(t,g)], sim["t%s_%s_px"%(t,g)])
            sim["m_%s_%s"%(g,t)] = np.sqrt(np.abs(sim["t%s_%s_e"%(t,g)]**2 - sim["t%s_%s_px"%(t,g)]**2\
                                         - sim["t%s_%s_py"%(t,g)]**2 - sim["t%s_%s_pz"%(t,g)]**2))
        if g == "rec":
            sim["met"] = np.sqrt(sim["met_%s_px"%g]**2 + sim["met_%s_py"%g]**2)
            sim["metphi"] = np.arctan2(sim["met_%s_py"%g], sim["met_%s_px"%g])

    for t in ["1", "2"]:
        for p in ["pt", "eta", "phi", "m"]:
            data["%s_rec_%s"%(p, t)] = data["%s_%s"%(p, t)]

    genbosonbranches = ["genbosonpt", "genbosoneta", "genbosonphi", "genbosonmass"]
    sim["genbosonpt"] = sim["pt_gen_b"]
    sim["genbosoneta"] = sim["eta_gen_b"]
    sim["genbosonphi"] = sim["phi_gen_b"]
    sim["genbosonmass"] = sim["m_gen_b"]

    # Plot
    for key in ["pt_rec_1", "pt_rec_2", "eta_rec_1", "eta_rec_2", "phi_rec_1", "phi_rec_2", "m_rec_1", "m_rec_2"] + branches + genbosonbranches:
        plt.figure(figsize=(6,6))
        q = np.percentile(data[key], [1, 99])
        _, bins, _ = plt.hist(data[key], histtype="step", lw=3, range=q, density=True, bins=20, label="2017")
        plt.hist(sim[key], histtype="step", lw=3, bins=bins, density=True, label="Own")
        plt.xlabel(key)
        plt.xlim(q)
        plt.legend()
        plt.savefig(key + "_c.png")


if __name__ == "__main__":
    main()
