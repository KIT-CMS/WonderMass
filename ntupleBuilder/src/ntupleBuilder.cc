// -*- C++ -*-
// //
// // Package:    ntupleBuilder
// // Class:      ntupleBuilder

#include <iostream>
#include <memory>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"


#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

template <typename T>
void subtractInvisible(reco::Candidate::LorentzVector &p4, T &x) {
  for (auto d = x->begin(); d != x->end(); d++) {
    const auto pdgId = d->pdgId();
    if (std::abs(pdgId) == 12 || std::abs(pdgId) == 14 ||
        std::abs(pdgId) == 16 || std::abs(pdgId) == 18) {
      p4 = p4 - d->p4();
    }
    subtractInvisible(p4, d);
  }
}

void AddP4Branch(TTree *tree, float *v, TString name) {
  tree->Branch(name + "_px", v + 0, name + "_px/F");
  tree->Branch(name + "_py", v + 1, name + "_py/F");
  tree->Branch(name + "_pz", v + 2, name + "_pz/F");
  tree->Branch(name + "_e", v + 3, name + "_e/F");
}

void SetP4Values(reco::Candidate::LorentzVector &p4, float *v) {
  v[0] = p4.px();
  v[1] = p4.py();
  v[2] = p4.pz();
  v[3] = p4.e();
}

template <typename T>
int FindTau(T taus, reco::Candidate::LorentzVector &p4, float min_dr) {
  int idx = -1;
  float dr = 100.0;
  for (auto tau = taus->begin(); tau != taus->end(); tau++) {
    const auto tmp = deltaR(p4, tau->p4());
    if (tmp < dr && tmp < min_dr) {
      dr = tmp;
      idx = tau - taus->begin();
    }
  }
  return idx;
}

class ntupleBuilder : public edm::EDAnalyzer {
public:
  explicit ntupleBuilder(const edm::ParameterSet &);
  ~ntupleBuilder();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event &, const edm::EventSetup &);
  virtual void endJob();
  bool providesGoodLumisection(const edm::Event &iEvent);
  bool isData;

  std::string processType;
  TTree *tree;

  // Event information
  Int_t v_run;
  UInt_t v_lumi_block;
  ULong64_t v_event;

  // Generator
  // Boson
  TLorentzVector lv_boson_gen;
  float v_h_gen[4];
  int v_h_gen_pdgid;
  int v_h_gen_process;
  float v_h_gen_mass;
  // tau
  TLorentzVector lv_tau1_gen, lv_tau2_gen;
  float v_t1_gen[4];
  float v_t2_gen[4];
  // visible tau
  TLorentzVector lv_vistau1_gen, lv_vistau2_gen;
  float v_t1_genvis[4];
  float v_t2_genvis[4];
  // MET
  TLorentzVector lv_met_gen;
  float v_met_gen[4];
  // Charge
  int v_t1_gen_q;
  int v_t2_gen_q;

  // Reco
  // visible tau
  TLorentzVector lv_tau1_rec, lv_tau2_rec;
  float v_t1_rec[4];
  float v_t2_rec[4];
  // Reco MET
  TLorentzVector lv_met_rec;
  float v_met_rec[2];
  // Decay modes
  int v_t1_rec_dm;
  int v_t2_rec_dm;
  // Charge
  int v_t1_rec_q;
  int v_t2_rec_q;

  // Tokens
  edm::EDGetTokenT<pat::TauCollection> t_taus;
  edm::EDGetTokenT<pat::METCollection> t_mets;
  edm::EDGetTokenT<reco::GenParticleCollection> t_gens;
};

ntupleBuilder::ntupleBuilder(const edm::ParameterSet &iConfig) {
  processType = iConfig.getParameter<std::string>("processType");

  edm::Service<TFileService> fs;

  tree = fs->make<TTree>("Events", "Events");

  // Event information
  tree->Branch("run", &v_run);
  tree->Branch("luminosityBlock", &v_lumi_block);
  tree->Branch("event", &v_event);

  // Four-vectors
  tree->Branch("lv_boson_gen", &lv_boson_gen);
  tree->Branch("lv_tau1_gen", &lv_tau1_gen);
  tree->Branch("lv_tau2_gen", &lv_tau2_gen);
  tree->Branch("lv_vistau1_gen", &lv_vistau1_gen);
  tree->Branch("lv_vistau2_gen", &lv_vistau2_gen);
  tree->Branch("lv_met_gen", &lv_met_gen);
  tree->Branch("lv_tau1_rec", &lv_tau1_rec);
  tree->Branch("lv_tau2_rec", &lv_tau2_rec);
  tree->Branch("lv_met_rec", &lv_met_rec);
  AddP4Branch(tree, v_h_gen, "h_gen");
  tree->Branch("h_gen_pdgid", &v_h_gen_pdgid, "h_gen_pdgid/I");
  tree->Branch("h_gen_mass", &v_h_gen_mass, "h_gen_mass/F");
  tree->Branch("h_gen_process", &v_h_gen_process, "h_gen_process/I");
  AddP4Branch(tree, v_t1_gen, "t1_gen");
  AddP4Branch(tree, v_t2_gen, "t2_gen");
  AddP4Branch(tree, v_t1_genvis, "t1_genvis");
  AddP4Branch(tree, v_t2_genvis, "t2_genvis");
  AddP4Branch(tree, v_met_gen, "met_gen");
  AddP4Branch(tree, v_t1_rec, "t1_rec");
  AddP4Branch(tree, v_t2_rec, "t2_rec");
  tree->Branch("met_rec_px", v_met_rec + 0, "met_rec_px/F");
  tree->Branch("met_rec_py", v_met_rec + 1, "met_rec_py/F");

  // Decay modes
  tree->Branch("t1_rec_dm", &v_t1_rec_dm, "t1_rec_dm/I");
  tree->Branch("t2_rec_dm", &v_t2_rec_dm, "t2_rec_dm/I");

  // Charge
  tree->Branch("t1_rec_q", &v_t1_rec_q, "t1_rec_q/I");
  tree->Branch("t2_rec_q", &v_t2_rec_q, "t2_rec_q/I");
  tree->Branch("t1_gen_q", &v_t1_gen_q, "t1_gen_q/I");
  tree->Branch("t2_gen_q", &v_t2_gen_q, "t2_gen_q/I");

  // Consumers
  t_taus =
      consumes<pat::TauCollection>(edm::InputTag("slimmedTaus", "", "PAT"));
  t_mets =
      consumes<pat::METCollection>(edm::InputTag("slimmedMETs", "", "PAT"));
  t_gens = consumes<reco::GenParticleCollection>(
      edm::InputTag("prunedGenParticles", "", "PAT"));
}

ntupleBuilder::~ntupleBuilder() {}

void ntupleBuilder::analyze(const edm::Event &iEvent,
                            const edm::EventSetup &iSetup) {

  // Event information
  // bool keepevent = true;
  v_run = iEvent.run();
  v_lumi_block = iEvent.luminosityBlock();
  v_event = iEvent.id().event();

  // GenParticles
  edm::Handle<reco::GenParticleCollection> gens;
  iEvent.getByToken(t_gens, gens);

  // TODO: https://github.com/cms-sw/cmssw/blob/master/RecoVertex/V0Producer/src/V0Fitter.cc#L49
  // https://github.com/greyxray/AOD_pi0_study/blob/master/AOD_pi0/python/ConfFileWithHPSTracks_cfg.py
  // const std::string processType = ""; // NOTE: To be replace before compilation by the job script.
  if (processType.compare("GGH") == 0) {
      v_h_gen_process = 0;
      v_h_gen_pdgid = 25;
  } else if (processType.compare("QQH") == 0) {
      v_h_gen_process = 1;
      v_h_gen_pdgid = 25;
  } else if ((processType.find("DY") != std::string::npos)) {
      v_h_gen_process = 2;
      v_h_gen_pdgid = 23;
  } else {
      throw std::runtime_error("Unknown process type encountered.");
  }

  std::vector<reco::GenParticle> genBosonCands;
  for (auto gen = gens->begin(); gen != gens->end(); gen++) {
    if (gen->pdgId() == v_h_gen_pdgid && gen->isLastCopy() == 1) {
      if (genBosonCands.size() != 0)
          std::cerr << "WARNING: Found more than one target boson with PDG id " << v_h_gen_pdgid << "!" << std::endl;
      genBosonCands.emplace_back(*gen);
    }
  }
  if (genBosonCands.size() != 1)
      return;

  // Get generator Higgs
  auto gen_boson = genBosonCands[0];
  auto gen_boson_p4 = gen_boson.p4();
  SetP4Values(gen_boson_p4, v_h_gen);
  v_h_gen_mass = gen_boson_p4.mass();
  v_h_gen_pdgid = gen_boson.pdgId();
  lv_boson_gen.SetPxPyPzE(gen_boson.p4().Px(),gen_boson.p4().Py(),gen_boson.p4().Pz(),gen_boson.p4().E());

  // Get generator taus
  if (gen_boson.numberOfDaughters() != 2)
    throw std::runtime_error("Failed to find two tau daughters.");
  auto gen_t1 = gen_boson.daughterRef(0);
  auto gen_t2 = gen_boson.daughterRef(1);
  if (gen_t1->pt() < gen_t2->pt()) { // Make taus pt ordered by gen pt
      const auto tmp = gen_t1;
      gen_t1 = gen_t2;
      gen_t2 = tmp;
  }
  auto t1_gen_p4 = gen_t1->p4();
  auto t2_gen_p4 = gen_t2->p4();
  SetP4Values(t1_gen_p4, v_t1_gen);
  SetP4Values(t2_gen_p4, v_t2_gen);
  lv_tau1_gen.SetPxPyPzE(t1_gen_p4.Px(),t1_gen_p4.Py(),t1_gen_p4.Pz(),t1_gen_p4.E());
  lv_tau2_gen.SetPxPyPzE(t2_gen_p4.Px(),t2_gen_p4.Py(),t2_gen_p4.Pz(),t2_gen_p4.E());
  v_t1_gen_q = gen_t1->charge();
  v_t2_gen_q = gen_t2->charge();

  if (v_t1_gen_q == v_t2_gen_q)
    throw std::runtime_error("Generator taus have same charge.");

  // Get four-vector of visible tau components
  auto t1_vis_p4 = t1_gen_p4;
  subtractInvisible(t1_vis_p4, gen_t1);
  if (t1_vis_p4 == t1_gen_p4)
  {
    if (std::abs(gen_t1->pdgId()) == 17 || std::abs(gen_t1->pdgId()) == 15) throw std::runtime_error("And it is a tau!");
    else
    {
      // keepevent = false;
      if (std::abs(gen_t1->pdgId()) != 11 && std::abs(gen_t1->pdgId()) != 13) std::cout << "Tau 1 does not have any neutrinos. pdgid: " << gen_t1->pdgId() << std::endl;
      return;
    }
  }
  SetP4Values(t1_vis_p4, v_t1_genvis);
  lv_vistau1_gen.SetPxPyPzE(t1_vis_p4.Px(),t1_vis_p4.Py(),t1_vis_p4.Pz(),t1_vis_p4.E());

  auto t2_vis_p4 = t2_gen_p4;
  subtractInvisible(t2_vis_p4, gen_t2);
  if (t2_vis_p4 == t2_gen_p4)
  {
    if (std::abs(gen_t2->pdgId()) == 17 || std::abs(gen_t1->pdgId()) == 15) throw std::runtime_error("And it is a tau!");
    else
    {
      // keepevent = false;
      if (std::abs(gen_t1->pdgId()) != 11 && std::abs(gen_t1->pdgId()) != 13) std::cout << "Tau 2 does not have any neutrinos. pdgid: " << gen_t2->pdgId() << std::endl;
      return;
    }
  }
  SetP4Values(t2_vis_p4, v_t2_genvis);
  lv_vistau2_gen.SetPxPyPzE(t2_vis_p4.Px(),t2_vis_p4.Py(),t2_vis_p4.Pz(),t2_vis_p4.E());

  // Reconstructed MET
  edm::Handle<pat::METCollection> mets;
  iEvent.getByToken(t_mets, mets);

  if (mets->size() != 1)
    throw std::runtime_error("Found no MET.");
  if (mets->at(0).isPFMET() == false)
    throw std::runtime_error("MET is no PFMet.");

  v_met_rec[0] = mets->at(0).corPx();
  v_met_rec[1] = mets->at(0).corPy();
  lv_met_rec.SetPxPyPzE(v_met_rec[0], v_met_rec[1], 0, 0);
  // Generator MET
  auto gen_met_p4 = mets->at(0).genMET()->p4();
  SetP4Values(gen_met_p4, v_met_gen);
  lv_met_gen.SetPxPyPzE(gen_met_p4.Px(),gen_met_p4.Py(),gen_met_p4.Pz(),gen_met_p4.E());

  // Reconstructed taus
  edm::Handle<pat::TauCollection> taus;
  iEvent.getByToken(t_taus, taus);

  // Ensure that we have two reconstructed taus
  if (taus->size() < 2)
    return;

  // Ensure that we can match both taus to different generator taus
  const float min_dr = 0.3; // Minimum deltaR valid for matching
  const auto idx1 = FindTau(taus, t1_vis_p4, min_dr);
  const auto idx2 = FindTau(taus, t2_vis_p4, min_dr);
  if (idx1 == -1 || idx2 == -1)
    return;
  if (idx1 == idx2)
    return;

  // Ensure that both taus have a reconstructed decay mode
  v_t1_rec_dm = taus->at(idx1).decayMode();
  v_t2_rec_dm = taus->at(idx2).decayMode();
  if (v_t1_rec_dm < 0 || v_t2_rec_dm < 0)
    return;

  // Ensure that the taus have opposite charge
  v_t1_rec_q = taus->at(idx1).charge();
  v_t2_rec_q = taus->at(idx2).charge();
  if (v_t1_rec_q == v_t2_rec_q)
    return;
  if (v_t1_rec_q == 0 || v_t2_rec_q == 0)
    return;

  // Fill four-vector
  auto t1_rec_p4 = taus->at(idx1).p4();
  auto t2_rec_p4 = taus->at(idx2).p4();
  SetP4Values(t1_rec_p4, v_t1_rec);
  SetP4Values(t2_rec_p4, v_t2_rec);
  lv_tau1_rec.SetPxPyPzE(t1_rec_p4.Px(),t1_rec_p4.Py(),t1_rec_p4.Pz(),t1_rec_p4.E());
  lv_tau2_rec.SetPxPyPzE(t2_rec_p4.Px(),t2_rec_p4.Py(),t2_rec_p4.Pz(),t2_rec_p4.E());

  // Apply baseline selection
  if (taus->at(idx1).pt() < 40) return;
  if (taus->at(idx2).pt() < 40) return;

  if (std::abs(taus->at(idx1).eta()) > 2.1) return;
  if (std::abs(taus->at(idx2).eta()) > 2.1) return;

  if (deltaR(t1_rec_p4, t2_rec_p4) < 0.5) return;

  const auto nameDM = "decayModeFinding";
  if (taus->at(idx1).tauID(nameDM) < 0.5) return;
  if (taus->at(idx2).tauID(nameDM) < 0.5) return;

  const auto nameIso = "byVLooseIsolationMVArun2v1DBoldDMwLT";
  if (taus->at(idx1).tauID(nameIso) < 0.5) return;
  if (taus->at(idx2).tauID(nameIso) < 0.5) return;

  // Fill event
  tree->Fill();
}

void ntupleBuilder::beginJob() {}

void ntupleBuilder::endJob() {}

DEFINE_FWK_MODULE(ntupleBuilder);
