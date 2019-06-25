// -*- C++ -*-
// //
// // Package:    ntupleBuilder
// // Class:      ntupleBuilder

#include <iostream>
#include <memory>
#include <algorithm>
#include <iterator>

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
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"


typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > RMDLV;

template<typename T1, typename T2>
inline void copyP4(const T1 &in, T2 &out)
{
  out.SetCoordinates(in.pt(), in.eta(), in.phi(), in.mass());
}

template <typename T>
void subtractInvisible(reco::Candidate::LorentzVector &p4, T &x)
{
  for (auto d = x->begin(); d != x->end(); d++) {
    const auto pdgId = d->pdgId();
    if (std::abs(pdgId) == 12 || std::abs(pdgId) == 14 ||
        std::abs(pdgId) == 16 || std::abs(pdgId) == 18)
    {
      p4 = p4 - d->p4();
    }
    subtractInvisible(p4, d);
  }
}


void AddP4Branch(TTree *tree, float *v, TString name)
{
  tree->Branch(name + "_px", v + 0, name + "_px/F");
  tree->Branch(name + "_py", v + 1, name + "_py/F");
  tree->Branch(name + "_pz", v + 2, name + "_pz/F");
  tree->Branch(name + "_e", v + 3, name + "_e/F");
  tree->Branch(name + "_pt", v + 4, name + "_pt/F");
  tree->Branch(name + "_eta", v + 5, name + "_eta/F");
  tree->Branch(name + "_phi", v + 6, name + "_phi/F");
}

void SetP4Values(reco::Candidate::LorentzVector &p4, float *v)
{
  v[0] = p4.px();
  v[1] = p4.py();
  v[2] = p4.pz();
  v[3] = p4.e();
  v[4] = p4.pt();
  v[5] = p4.eta();
  v[6] = p4.phi();
}

template <typename T>
int FindTau(T taus, reco::Candidate::LorentzVector &p4, float min_dr)
{
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


template <typename T>
int CountJets(T jets, reco::Candidate::LorentzVector & l1_p4, float min_dr1, reco::Candidate::LorentzVector & l2_p4, float min_dr2)
{
  int count = 0;
  for (auto jet = jets->begin(); jet != jets->end(); jet++)
  {
    if (jet->p4().pt() <= 20 || abs(jet->p4().eta()) > 4.7 ) continue;
    if (deltaR(l1_p4, jet->p4()) <= min_dr1 || deltaR(l2_p4, jet->p4()) <= min_dr2) continue;
    // Tight ID
    // TODO:  , do universal ; ID: tight PF Jet ID (https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVRun2017#Preliminary_Recommendations_for)
      if (std::abs(jet->p4().eta()) <= 2.4f)
      {
        if (!((jet->neutralHadronEnergyFraction() < 0.90f)
                && (jet->photonEnergyFraction() + jet->hoEnergyFraction() < 0.90f) // Neutral EM Fraction
                && (jet->nConstituents() > 1)
                && (jet->chargedHadronEnergyFraction() > 0.0)
                // && (jet->chargedMuEnergyFraction() < 80)
                && (jet->chargedHadronMultiplicity() > 0))) continue;
      }
      else if (std::abs(jet->p4().eta()) > 2.4f && std::abs(jet->p4().eta()) <= 2.7f)
      {
        if (!((jet->neutralHadronEnergyFraction() < 0.90f)
                && (jet->photonEnergyFraction() + jet->hoEnergyFraction() < 0.90f) // Neutral EM Fraction
                && (jet->nConstituents() > 1))) continue;
      }
      else if (std::abs(jet->p4().eta()) > 2.7f && std::abs(jet->p4().eta()) <= 3.0f)
      {
        if (!((jet->photonEnergyFraction() + jet->hoEnergyFraction() > 0.02f)
                && (jet->photonEnergyFraction() + jet->hoEnergyFraction() < 0.99f) // Neutral EM Fraction
                && (jet->nConstituents() - jet->chargedHadronMultiplicity() > 2))) continue;
      }
      else if (std::abs(jet->p4().eta()) > 3.0f)
      {
        if (!((jet->neutralHadronEnergyFraction() > 0.02f)
                && (jet->photonEnergyFraction() + jet->hoEnergyFraction() < 0.90f) // Neutral EM Fraction
                && (jet->nConstituents() - jet->chargedHadronMultiplicity() > 10))) continue; // Number of neutral particles
      }
    count++;
  }

  return count;
}

class ntupleBuilder : public edm::EDAnalyzer
{
  public:
    explicit ntupleBuilder(const edm::ParameterSet &);
    ~ntupleBuilder();

  private:
    virtual void beginJob();
    virtual void analyze(const edm::Event &, const edm::EventSetup &);
    virtual void endJob();
    bool providesGoodLumisection(const edm::Event &iEvent);
    bool isData;
    bool return_flag;

    std::string processType;
    bool debug;
    bool baseline;
    TTree *tree;
    TTree *tree_tt;
    TTree *tree_mt;
    TTree *tree_et;
    TTree *tree_gen;
    TH1I *cutflow;
    static const Int_t nx = 8;
    const char *cuts[nx] = {
      "genBosonCands!=1",  // 0
      "tau no nu", // 1
      "recoTau < 2", // 2
      "no genTau match or same", // 3
      "no reco DM", // 4
      "same charge or 0 charge", // 5
      "baseline", // 6
      "Z->tautau" // 7
    };
    int gen_z_decay;
    int gen_tau_decay;
    int gen_tau1_decay;
    int gen_tau2_decay;
    // static const Int_t n_gen_z_decay = 3;
    // const char *gen_z_decay[n_gen_z_decay] = {
    //   "Z->ee",  // 0
    //   "Z->mumu", // 1
    //   "Z->tautau", // 2
    // };
    // static const Int_t n_gen_tau_decay = 6;
    // const char *gen_tau_decay[n_gen_tau_decay] = {
    //   "tau->e",  // 0
    //   "tau->mu", // 1
    //   "tau_h->1pi",  // 2
    //   "tau_h->1piNpi0", // 3
    //   "tau_h->3pi", // 4
    //   "tau_h->3piNpio", // 5
    // };
    // TTree *tree_cutflow;
    // int cutflow;

    // Event information
    Int_t v_run;
    UInt_t v_lumi_block;
    ULong64_t v_event;

    // Generator
    // Boson
    TLorentzVector lv_boson_gen;
    float v_h_gen[7];
    int v_h_gen_pdgid;
    int v_h_gen_process;
    float v_h_gen_mass;
    // tau
    TLorentzVector lv_tau1_gen, lv_tau2_gen;
    float v_t1_gen[7];
    float v_t2_gen[7];
    // visible tau
    TLorentzVector lv_vistau1_gen, lv_vistau2_gen;
    float v_t1_genvis[7];
    float v_t2_genvis[7];
    // MET
    TLorentzVector lv_met_gen;
    float v_met_gen[7];
    // Charge
    int t1_gen_q;
    int t2_gen_q;
    int t1_gen_flav;
    int t2_gen_flav;
    int t2_gen_dm;

    // Generator
    int expected_rec_channel;
    // Boson
    TLorentzVector lv_boson_trgen;
    float v_h_trgen[7];
    int v_h_trgen_pdgid;
    int v_h_trgen_process;
    float v_h_trgen_mass;
    // tau
    TLorentzVector lv_tau1_trgen, lv_tau2_trgen;
    float v_t1_trgen[7];
    float v_t2_trgen[7];
    // visible tau
    TLorentzVector lv_vistau1_trgen, lv_vistau2_trgen;
    float v_t1_trgenvis[7];
    float v_t2_trgenvis[7];
    // MET
    TLorentzVector lv_met_trgen;
    float v_met_trgen[7];
    // Charge
    int t1_trgen_q;
    int t2_trgen_q;
    int t1_trgen_flav;
    int t2_trgen_flav;

    // Reco
    int channel_rec;
    // visible tau
    TLorentzVector lv_tau1_rec, lv_tau2_rec;
    float v_t1_rec[7];
    float v_t2_rec[7];
    // Reco MET
    TLorentzVector lv_met_rec;
    float v_met_rec[2];
    // Decay modes
    int v_t1_rec_dm;
    int v_t2_rec_dm;
    // Charge
    int v_t1_rec_q;
    int v_t2_rec_q;
    int njets;

    // Tokens
    edm::EDGetTokenT<pat::TauCollection> t_taus;
    edm::EDGetTokenT<pat::MuonCollection> t_muons;
    edm::EDGetTokenT<pat::ElectronCollection> t_electrons;
    edm::EDGetTokenT<pat::JetCollection> t_jets;
    edm::EDGetTokenT<reco::VertexCollection> t_vtx;

    edm::EDGetTokenT<pat::METCollection> t_mets;
    edm::EDGetTokenT<reco::GenParticleCollection> t_gens;

    void AddTreeBranches(TTree *tree)
    {
      // Event information
      tree->Branch("run", &v_run);
      tree->Branch("luminosityBlock", &v_lumi_block);
      tree->Branch("event", &v_event);
      // Event tree
      //// Gen
      tree->Branch("expected_rec_channel", &expected_rec_channel, "expected_rec_channel/I");
      tree->Branch("t1_gen_q", &t1_gen_q, "t1_gen_q/I");
      tree->Branch("t2_gen_q", &t2_gen_q, "t2_gen_q/I");
      tree->Branch("t1_gen_flav", &t1_gen_flav, "t1_gen_flav/I");
      tree->Branch("t2_gen_flav", &t2_gen_flav, "t2_gen_flav/I");
      tree->Branch("lv_boson_gen", &lv_boson_gen);
      tree->Branch("lv_tau1_gen", &lv_tau1_gen);
      tree->Branch("lv_tau2_gen", &lv_tau2_gen);
      tree->Branch("lv_vistau1_gen", &lv_vistau1_gen);
      tree->Branch("lv_vistau2_gen", &lv_vistau2_gen);
      tree->Branch("lv_met_gen", &lv_met_gen);
      AddP4Branch(tree, v_t1_gen, "t1_gen");
      AddP4Branch(tree, v_t2_gen, "t2_gen");
      AddP4Branch(tree, v_t1_genvis, "t1_genvis");
      AddP4Branch(tree, v_t2_genvis, "t2_genvis");
      AddP4Branch(tree, v_met_gen, "met_gen");
      AddP4Branch(tree, v_h_gen, "h_gen");
      tree->Branch("h_gen_pdgid", &v_h_gen_pdgid, "h_gen_pdgid/I");
      tree->Branch("h_gen_mass", &v_h_gen_mass, "h_gen_mass/F");
      tree->Branch("h_gen_process", &v_h_gen_process, "h_gen_process/I");
      //// Rec
      tree->Branch("channel_rec", &channel_rec, "channel_rec/I");
      tree->Branch("lv_tau1_rec", &lv_tau1_rec);
      tree->Branch("lv_tau2_rec", &lv_tau2_rec);
      tree->Branch("lv_met_rec", &lv_met_rec);
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
      // Jets
      tree->Branch("njets", &njets, "njets/I");
    }

    struct DecayInfo
    {
      // mode is hadronic as long as we do not find an electron or muon in the
      // first level of the decay chain
      DecayInfo(): p4_vis(0,0,0,0), mode(Hadronic), n_charged(0) {}

      RMDLV p4_vis;

      enum { Electronic, Muonic, Hadronic } mode;
      unsigned int n_charged;
    };

    void walkDecayTree(const reco::GenParticle& in, DecayInfo& info, int level=0, bool allowNonPromptTauDecayProduct=false )
    {
      std::string level_string = "";
      for(int i = 0; i < level; i++) level_string += "\t";
      //for(int i = 0; i < level; ++i) printf(" ");
      //printf("PDG %d\tstatus %d", in.pdgId(), in.status());
      //std::cout<<"\tpt "<<in.p4().pt()<<"\tphi "<<in.p4().phi()<<"\teta "<<in.p4().eta()<<"\tE "<<in.p4().E()
      //<<"\tispromptaudecyp:"<<in.statusFlags().isPromptTauDecayProduct();

      unsigned int lep_daughter_wiht_max_pt = 0;

      RMDLV p4(0,0,0,0);
      if(in.numberOfDaughters() == 0)
      {
        //printf("\n");
        // Check that the final particle of the chain is a direct tau decay product, avoiding,
        // for example, particles coming from intermediate gamma emission, which are listed as
        // tau daughters in prunedGenParticle collections. Method available only from 74X.
        if(in.status() == 1 && !isNeutrino(in.pdgId()) && (in.statusFlags().isPromptTauDecayProduct() || allowNonPromptTauDecayProduct))
        {
          RMDLV p4;
          copyP4(in.p4(), p4);
          info.p4_vis += p4;

          if(std::abs(in.pdgId()) == 11 && std::abs(in.mother()->pdgId()) == 15) info.mode = DecayInfo::Electronic;
          if(std::abs(in.pdgId()) == 13 && std::abs(in.mother()->pdgId()) == 15) info.mode = DecayInfo::Muonic;

          if(in.charge() != 0) ++info.n_charged;
                                  //std::cout << "\t"+level_string+"Increasing number of charged decay products: " << info.n_charged << " because of particle with pdgId: " << in.pdgId()  << " and p4 = (" << in.pt() << "," << in.eta() << "," << in.phi() << "," << in.mass() << ")" << " at level " << level << std::endl;
        }
      }
      else if(in.numberOfDaughters() == 1)
      {
        //std::cout << "\t"+level_string+"one child, keeping level... \n";
        // Don't increase level since this does not seem to be a "real"
        // decay but just an intermediate generator step
        walkDecayTree(static_cast<const reco::GenParticle&>(*in.daughter(0)), info, level, allowNonPromptTauDecayProduct);
      }
      else if(in.numberOfDaughters() == 2 && (
          (std::abs(in.daughter(0)->pdgId()) == 22 && std::abs(in.daughter(1)->pdgId()) == 15) ||
          (std::abs(in.daughter(0)->pdgId()) == 15 && std::abs(in.daughter(1)->pdgId()) == 22))
           )
      {
        //std::cout << "\t"+level_string+"interm. gamma emission, keeping level... \n";
        // Don't increase level since this does not seem to be a "real"
        // decay but just an intermediate emission of a photon
        // Don't follow photon decay path
        if (std::abs(in.daughter(0)->pdgId()) == 15)
          walkDecayTree(dynamic_cast<const reco::GenParticle&>(*in.daughter(0)), info, level, allowNonPromptTauDecayProduct);
        else
          walkDecayTree(dynamic_cast<const reco::GenParticle&>(*in.daughter(1)), info, level, allowNonPromptTauDecayProduct);
      }
      //else if(in.numberOfDaughters() >= 3 && ( isLepton(in.daughter(0)->pdgId()) &&  isLepton(in.daughter(1)->pdgId()) && isLepton(in.daughter(2)->pdgId())))
      else if (minthreeLeptondauhhters(in, lep_daughter_wiht_max_pt))
      {
        //std::cout << "\t"+level_string+"internal gamma(*) conversion (tau -> l l tau )\n";
        // Don't increase level since the initial tau is not real.
        // The inital tau decays into three leptons, which can be illustrated by a virual photon (gamma*).
        // Take the lepton with the largest pt, which is most likly to be reconstructed as the tau. The others are usally soft
        // the loop over the daughters is necessary since also photons can be radiated within this step.
        if (std::abs(in.daughter(lep_daughter_wiht_max_pt)->pdgId()) == 11 || std::abs(in.daughter(lep_daughter_wiht_max_pt)->pdgId()) == 13)
          allowNonPromptTauDecayProduct = true;
         walkDecayTree(dynamic_cast<const reco::GenParticle&>(*in.daughter(lep_daughter_wiht_max_pt)), info, level, allowNonPromptTauDecayProduct);
      }
      else if (minThreeDaughtersWithQQBar(in, lep_daughter_wiht_max_pt))
      {
        //std::cout << "\t"+level_string+"internal gamma(*) conversion (tau -> q q tau )\n";
        // Don't increase level since the initial tau is not real.
        // The inital tau decays into three leptons, which can be illustrated by a virual photon (gamma*).
        // Take the lepton with the largest pt, which is most likly to be reconstructed as the tau. The others are usally soft
        // the loop over the daughters is necessary since also photons can be radiated within this step.
        if (std::abs(in.daughter(lep_daughter_wiht_max_pt)->pdgId()) == 11 || std::abs(in.daughter(lep_daughter_wiht_max_pt)->pdgId()) == 13)
                                  allowNonPromptTauDecayProduct = true;
              walkDecayTree(dynamic_cast<const reco::GenParticle&>(*in.daughter(lep_daughter_wiht_max_pt)), info, level, allowNonPromptTauDecayProduct);
      }
      else if(in.numberOfDaughters() == 2 && std::abs(in.pdgId()) == 111 &&
          std::abs(in.daughter(0)->pdgId()) == 22 && std::abs(in.daughter(1)->pdgId()) == 22)
      {
        //std::cout << "\t"+level_string+"neutral pion, stop recursion. \n";
        //printf("\n");
        // Neutral pion, save four-momentum in the visible component
        RMDLV p4;
        copyP4(in.p4(), p4);
        info.p4_vis += p4;
      }
      else if(in.numberOfDaughters() == 3 && std::abs(in.pdgId()) == 111 &&
          ((std::abs(in.daughter(0)->pdgId()) == 22 && std::abs(in.daughter(1)->pdgId()) == 11 && std::abs(in.daughter(2)->pdgId()) == 11) ||
          (std::abs(in.daughter(0)->pdgId()) == 11 && std::abs(in.daughter(1)->pdgId()) == 22 && std::abs(in.daughter(2)->pdgId()) == 11) ||
          (std::abs(in.daughter(0)->pdgId()) == 11 && std::abs(in.daughter(1)->pdgId()) == 11 && std::abs(in.daughter(2)->pdgId()) == 22)))
      {
        //std::cout << "\t"+level_string+"neutral pion, stop recursion. \n";
        //printf("\n");
        // Neutral pion, save four-momentum in the visible component
        RMDLV p4;
        copyP4(in.p4(), p4);
        info.p4_vis += p4;
      }
      else if(in.numberOfDaughters() == 4 && std::abs(in.pdgId()) == 111 &&
          std::abs(in.daughter(0)->pdgId()) == 11 && std::abs(in.daughter(1)->pdgId()) == 11 && std::abs(in.daughter(2)->pdgId()) == 11 && std::abs(in.daughter(3)->pdgId()) == 11)
      {
        //std::cout << "\t"+level_string+"neutral pion, stop recursion. \n";
        //printf("\n");
        // Neutral pion, save four-momentum in the visible component
        RMDLV p4;
        copyP4(in.p4(), p4);
        info.p4_vis += p4;
      }
      else
      {
        //std::cout << "\t"+level_string << in.numberOfDaughters() << " children, recurse...\n";
        for(unsigned int i = 0; i < in.numberOfDaughters(); ++i)
                          {
                                  //std::cout << "\t"+level_string+"Running recursion for " << (*in.daughter(i)).pdgId() << " with p4 " << (*in.daughter(i)).pt() << "," << (*in.daughter(i)).eta() << "," << (*in.daughter(i)).phi() << "," << (*in.daughter(i)).mass() << std::endl;
          walkDecayTree(dynamic_cast<const reco::GenParticle&>(*in.daughter(i)), info, level + 1, allowNonPromptTauDecayProduct);
                          }
      }
    }

    static bool isNeutrino(int pdg_id)
    {
      return std::abs(pdg_id) == 12 || std::abs(pdg_id) == 14 || std::abs(pdg_id) == 16;
    }

    static bool isLepton(int pdg_id)
    {
      return std::abs(pdg_id) == 11 || std::abs(pdg_id) == 13 || std::abs(pdg_id) == 15;
    }
    static bool isQuark(int pdg_id)
    {
      return std::abs(pdg_id) >= 1  && std::abs(pdg_id) <=  6;
    }

    // returns True if >=3 leptons as daughters and saves the number, corresponding to the daughter-lepton with highest p_T
    static bool minthreeLeptondauhhters(const reco::GenParticle& in, unsigned int &lep_daughter_wiht_max_pt)
    {
      int Nakt = 0;
      float akt_max_pt = -1.0;
      for(unsigned int i = 0; i < in.numberOfDaughters(); ++i)
      {
        if (isLepton(in.daughter(i)->pdgId()))
        {
          Nakt++;
          if (in.daughter(i)->pt() > akt_max_pt)
          {
            lep_daughter_wiht_max_pt = i;
            akt_max_pt = in.daughter(i)->pt();
          }
        }
      }
      return Nakt>=3;
    }

    static bool minThreeDaughtersWithQQBar(const reco::GenParticle& in, unsigned int &lep_daughter_wiht_max_pt)
    {
      int qqBar = 0;
                  int Nakt = 0;
      float akt_max_pt = -1.0;
      for(unsigned int i = 0; i < in.numberOfDaughters(); ++i)
      {
        if (isLepton(in.daughter(i)->pdgId()))
        {
          Nakt++;
          if (in.daughter(i)->pt() > akt_max_pt)
          {
            lep_daughter_wiht_max_pt = i;
            akt_max_pt = in.daughter(i)->pt();
          }
        }
                          else if (isQuark(in.daughter(i)->pdgId()))
                          {
                                  qqBar++;
                          }
      }
      return Nakt>=1 && qqBar == 2;
    }

    DecayInfo getGen(const reco::GenParticle& gen_tau, int *gen_tau_decay, std::string name)
    {

      // std::cout << name << ":" << gen_tau.pdgId() << " :";
      // for (reco::GenParticleRefVector::const_iterator  gen = gen_tau.daughterRefVector().begin(); gen != gen_tau.daughterRefVector().end(); gen++)
      // {
      //   std::cout << " " << (*gen)->pdgId() << ";";
      // }
      // std::cout << "\n";

      DecayInfo info;
      walkDecayTree(dynamic_cast<const reco::GenParticle&>(gen_tau), info);
      switch(info.mode)
      {
        case DecayInfo::Electronic:
          *gen_tau_decay = 1;
          return info;
          break;
        case DecayInfo::Muonic:
          *gen_tau_decay = 2;
          return info;
          break;
        case DecayInfo::Hadronic:
          if(info.n_charged == 1)
            *gen_tau_decay = 3;
          else if(info.n_charged == 3)
            *gen_tau_decay = 4;
          else{
            *gen_tau_decay = 5;
          }
          // TODO: Can this happen?
          return info;
          break;
        default:
          assert(false);
          return info;
          break;
      }
    }

};

ntupleBuilder::ntupleBuilder(const edm::ParameterSet &iConfig)
{
  processType = iConfig.getParameter<std::string>("processType");
  debug = iConfig.getParameter<bool>("debug");
  baseline = iConfig.getParameter<bool>("debug");

  edm::Service<TFileService> fs;

  tree = fs->make<TTree>("Events", "Events");
  tree_et = fs->make<TTree>("Events_et", "Events_et");
  tree_mt = fs->make<TTree>("Events_mt", "Events_mt");
  tree_tt = fs->make<TTree>("Events_tt", "Events_tt");
  tree_gen = fs->make<TTree>("GenEvents", "GenEvents");
  cutflow = fs->make<TH1I>( "cutflow"  , "cutflow", nx,  0, nx );
  for (int i = 1; i <= nx; i++) cutflow->GetXaxis()->SetBinLabel(i, cuts[i - 1]);


  // // Gen tree
  tree_gen->Branch("gen_z_decay", &gen_z_decay, "gen_z_decay/I");
  tree_gen->Branch("gen_tau1_decay", &gen_tau1_decay, "gen_tau1_decay/I");
  tree_gen->Branch("gen_tau2_decay", &gen_tau2_decay, "gen_tau2_decay/I");
  // tree_gen->Branch("gen_tau_decay", &gen_tau_decay, "gen_tau_decay/I");
  //   tree_gen->Branch("gen_z_decay", &gen_z_decay);
  //   tree_gen->Branch("gen_tau_decay", &gen_tau_decay);
  tree_gen->Branch("t1_gen_q", &t1_trgen_q, "t1_gen_q/I");
  tree_gen->Branch("t2_gen_q", &t2_trgen_q, "t2_gen_q/I");
  tree_gen->Branch("t1_gen_flav", &t1_trgen_flav, "t1_gen_flav/I");
  tree_gen->Branch("t2_gen_flav", &t2_trgen_flav, "t2_gen_flav/I");
  tree_gen->Branch("lv_boson_gen", &lv_boson_trgen);
  tree_gen->Branch("lv_tau1_gen", &lv_tau1_trgen);
  tree_gen->Branch("lv_tau2_gen", &lv_tau2_trgen);
  tree_gen->Branch("lv_vistau1_gen", &lv_vistau1_trgen);
  tree_gen->Branch("lv_vistau2_gen", &lv_vistau2_trgen);
  tree_gen->Branch("lv_met_gen", &lv_met_trgen);
  AddP4Branch(tree_gen, v_t1_trgen, "t1_gen");
  AddP4Branch(tree_gen, v_t2_trgen, "t2_gen");
  AddP4Branch(tree_gen, v_t1_trgenvis, "t1_genvis");
  AddP4Branch(tree_gen, v_t2_trgenvis, "t2_genvis");
  AddP4Branch(tree_gen, v_met_trgen, "met_gen");
  AddP4Branch(tree_gen, v_h_trgen, "h_gen");
  tree_gen->Branch("expected_rec_channel", &expected_rec_channel, "expected_rec_channel/I");
  tree_gen->Branch("h_gen_pdgid", &v_h_trgen_pdgid, "h_gen_pdgid/I");
  tree_gen->Branch("h_gen_mass", &v_h_trgen_mass, "h_gen_mass/F");
  tree_gen->Branch("h_gen_process", &v_h_trgen_process, "h_gen_process/I");

  // tree_cutflow = fs->make<TTree>("Cutflow", "Cutflow");
  // tree_cutflow->Branch("cutflow", &cutflow, "cutflow/I");

  AddTreeBranches(tree);
  AddTreeBranches(tree_tt);
  AddTreeBranches(tree_mt);
  AddTreeBranches(tree_et);

  // Consumers
  t_taus = consumes<pat::TauCollection>(edm::InputTag("slimmedTaus", "", "PAT"));
  t_muons = consumes<pat::MuonCollection>(edm::InputTag("slimmedMuons", "", "PAT"));
  t_electrons = consumes<pat::ElectronCollection>(edm::InputTag("slimmedElectrons", "", "PAT"));
  t_jets = consumes<pat::JetCollection>(edm::InputTag("slimmedJets", "", "PAT"));
  t_mets = consumes<pat::METCollection>(edm::InputTag("slimmedMETs", "", "PAT"));
  t_vtx = consumes<reco::VertexCollection>(edm::InputTag("offlineSlimmedPrimaryVertices", "", "PAT"));

  t_gens = consumes<reco::GenParticleCollection>(edm::InputTag("prunedGenParticles", "", "PAT"));
}

ntupleBuilder::~ntupleBuilder() {}

void ntupleBuilder::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
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
  std::vector<int> allowed_bosons;
  if (processType.find("GGH") != std::string::npos) {
      v_h_gen_process = 0;
      // v_h_gen_pdgid = 25;
      allowed_bosons.push_back(25);
      allowed_bosons.push_back(35);
      allowed_bosons.push_back(36);
  } else if (processType.find("QQH") != std::string::npos) {
      v_h_gen_process = 1;
      // v_h_gen_pdgid = 25;
      allowed_bosons.push_back(25);
      allowed_bosons.push_back(35);
      allowed_bosons.push_back(36);
      // allowed_bosons.insert(allowed_bosons.end(), a, a+5);
  } else if ((processType.find("DY") != std::string::npos)) {
      v_h_gen_process = 2;
      // v_h_gen_pdgid = 23;

      allowed_bosons.push_back(23);
  } else {
      throw std::runtime_error("Unknown process type encountered.");
  }

  std::vector<reco::GenParticle> genBosonCands;
  for (auto gen = gens->begin(); gen != gens->end(); gen++)
  {
    // if (gen->pdgId() == v_h_gen_pdgid && gen->isLastCopy() == 1)
    if ((std::find(allowed_bosons.begin(), allowed_bosons.end(), std::abs(gen->pdgId())) != allowed_bosons.end()) && gen->isLastCopy() == 1)
    {
      // if (debug) std::cout << "gen->pdgId(): " << gen->pdgId() << "  gen->isLastCopy():" <<  gen->isLastCopy() << " \n";
      if (genBosonCands.size() != 0)
          std::cerr << "WARNING: Found more than one target boson with PDG id " << gen->pdgId() << "!" << std::endl;
      genBosonCands.emplace_back(*gen);
    }
  }
  if (genBosonCands.size() != 1)
  {
      if (debug) std::cout << "genBosonCands.size(): " << genBosonCands.size() << " != 1 \n";
      cutflow->Fill(0);
      return;
  }

  // Get generator Higgs
  auto gen_boson = genBosonCands[0];
  auto gen_boson_p4 = gen_boson.p4();
  SetP4Values(gen_boson_p4, v_h_gen);
  v_h_gen_mass = gen_boson_p4.mass();
  v_h_gen_pdgid = gen_boson.pdgId();
  lv_boson_gen.SetPxPyPzE(gen_boson.p4().Px(),gen_boson.p4().Py(),gen_boson.p4().Pz(),gen_boson.p4().E());

  // Get generator taus
  if (gen_boson.numberOfDaughters() != 2)
    throw std::runtime_error("Failed to find two daughters.");
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
  t1_gen_q = gen_t1->charge();
  t2_gen_q = gen_t2->charge();
  t1_gen_flav = gen_t1->pdgId();
  t2_gen_flav = gen_t2->pdgId();

  if (t1_gen_q == t2_gen_q) throw std::runtime_error("Generator taus have same charge.");

  // Get four-vector of visible tau components
  return_flag = false;
  if (std::abs(gen_t1->pdgId()) != 15 || std::abs(gen_t2->pdgId()) != 15)
  {
    // std::cout << "Not Z->tautau: " << gen_t1->pdgId() << " " << gen_t2->pdgId() << std::endl;
    cutflow->Fill(7);
    return;
  }
  auto t1_vis_p4 = t1_gen_p4;
  subtractInvisible(t1_vis_p4, gen_t1);
  if (t1_vis_p4 == t1_gen_p4)
  {
    if (std::abs(gen_t1->pdgId()) == 15) throw std::runtime_error("And it is a tau!");
    else
    {
      if (std::abs(gen_t1->pdgId()) != 11 && std::abs(gen_t1->pdgId()) != 13) std::cout << "Tau_h 1 does not have any neutrinos. pdgid: " << gen_t1->pdgId() << std::endl;
      cutflow->Fill(1);
      return_flag = true;
    }
  }
  SetP4Values(t1_vis_p4, v_t1_genvis);
  lv_vistau1_gen.SetPxPyPzE(t1_vis_p4.Px(),t1_vis_p4.Py(),t1_vis_p4.Pz(),t1_vis_p4.E());

  auto t2_vis_p4 = t2_gen_p4;
  subtractInvisible(t2_vis_p4, gen_t2);
  if (t2_vis_p4 == t2_gen_p4)
  {
    if (std::abs(gen_t2->pdgId()) == 15) throw std::runtime_error("And it is a tau!");
    else
    {
      if (std::abs(gen_t2->pdgId()) != 11 && std::abs(gen_t2->pdgId()) != 13) std::cout << "Tau_h 2 does not have any neutrinos. pdgid: " << gen_t2->pdgId() << std::endl;
      cutflow->Fill(1);
      return_flag = true;
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
    lv_met_gen.SetPxPyPzE(gen_met_p4.Px(), gen_met_p4.Py(), gen_met_p4.Pz(), gen_met_p4.E());

  if (std::abs(gen_t1->pdgId()) == 11 && std::abs(gen_t2->pdgId()) == 11) gen_z_decay = 1; // Z->ee
  else if (std::abs(gen_t1->pdgId()) == 13 && std::abs(gen_t2->pdgId()) == 13) gen_z_decay = 2; // Z->mumu
  else if (std::abs(gen_t1->pdgId()) == 15 && std::abs(gen_t2->pdgId()) == 15) gen_z_decay = 3; // Z->tautau
  else throw std::runtime_error("Unknown boson decay: gen_t1->pdgId()=" + std::to_string(gen_t1->pdgId()) + "; gen_t2->pdgId()=" + std::to_string(gen_t2->pdgId()));

  gen_tau1_decay = -1;
  gen_tau2_decay = -1;
  DecayInfo t1_p4_vis = getGen(dynamic_cast<const reco::GenParticle&>(*gen_t1), &gen_tau1_decay, "tau1");
  DecayInfo t2_p4_vis = getGen(dynamic_cast<const reco::GenParticle&>(*gen_t2), &gen_tau2_decay, "tau2");
  // Find which channel is it on gen level
  expected_rec_channel = -1;
  if (gen_tau1_decay == 1 && gen_tau2_decay > 2) // gen et
  {
    expected_rec_channel = 1;
  }
  else if (gen_tau1_decay > 2 && gen_tau2_decay == 1) // gen et, switch lepton order
  {
    expected_rec_channel = 1;

    std::swap(gen_t1, gen_t2);
    std::swap(t1_gen_p4, t2_gen_p4);
    std::swap(t1_gen_q, t2_gen_q);
    std::swap(t1_gen_flav, t2_gen_flav);
    std::swap(t1_vis_p4, t2_vis_p4);
    std::swap(gen_tau1_decay, gen_tau2_decay);

    std::swap(lv_tau1_gen, lv_tau2_gen);
    std::swap(lv_vistau1_gen, lv_vistau2_gen);
    std::swap(v_t1_gen, v_t2_gen);
    std::swap(v_t1_genvis, v_t2_genvis);

    std::swap(t1_p4_vis, t2_p4_vis);
  }
  else if (gen_tau1_decay == 2 && gen_tau2_decay > 2) // gen mt
  {
    expected_rec_channel = 2;
  }
  else if (gen_tau1_decay > 2 && gen_tau2_decay == 2) // gen mt, switch lepton order
  {
    expected_rec_channel = 2;
    std::swap(gen_t1, gen_t2);
    std::swap(t1_gen_p4, t2_gen_p4);
    std::swap(t1_gen_q, t2_gen_q);
    std::swap(t1_gen_flav, t2_gen_flav);
    std::swap(t1_vis_p4, t2_vis_p4);
    std::swap(gen_tau1_decay, gen_tau2_decay);

    std::swap(lv_tau1_gen, lv_tau2_gen);
    std::swap(lv_vistau1_gen, lv_vistau2_gen);
    std::swap(v_t1_gen, v_t2_gen);
    std::swap(v_t1_genvis, v_t2_genvis);

    std::swap(t1_p4_vis, t2_p4_vis);
  }
  else if (gen_tau1_decay > 2 && gen_tau2_decay > 2) // get tt
  {
    expected_rec_channel = 3;
  }
  else if ((gen_tau1_decay == 2 && gen_tau2_decay == 1) || (gen_tau1_decay == 1 && gen_tau2_decay == 2))
  {
    std::cout << "Skip em\n";
    expected_rec_channel = -1;
  }
  else if ((gen_tau1_decay == gen_tau2_decay ) && (gen_tau1_decay == 1 || gen_tau1_decay == 2))
  {
    std::cout << "Skip ee, mm\n";
    expected_rec_channel = -1;
  }
  else
  {
    std::cout << "Couldn't resolve real channel:\n";
    std::cout << "gen_tau1_decay:" << gen_tau1_decay << "\n";
    std::cout << "gen_tau2_decay:" << gen_tau2_decay << "\n";
    assert(false);
  }
  t1_trgen_q = t1_gen_q;
  t2_trgen_q = t2_gen_q;
  t1_trgen_flav = t1_gen_flav;
  t2_trgen_flav = t2_gen_flav;
  lv_boson_trgen = lv_boson_gen;
  lv_tau1_trgen = lv_tau1_gen;
  lv_tau2_trgen = lv_tau2_gen;
  lv_vistau1_trgen = lv_vistau1_gen;
  lv_vistau2_trgen = lv_vistau2_gen;
  lv_met_trgen = lv_met_gen;
  std::copy(std::begin(v_t1_gen), std::end(v_t1_gen), std::begin(v_t1_trgen));
  std::copy(std::begin(v_t2_gen), std::end(v_t2_gen), std::begin(v_t2_trgen));
  std::copy(std::begin(v_t1_genvis), std::end(v_t1_genvis), std::begin(v_t1_trgenvis));
  std::copy(std::begin(v_t2_genvis), std::end(v_t2_genvis), std::begin(v_t2_trgenvis));
  std::copy(std::begin(v_met_gen), std::end(v_met_gen), std::begin(v_met_trgen));
  std::copy(std::begin(v_h_gen), std::end(v_h_gen), std::begin(v_h_trgen));
  v_h_trgen_pdgid = v_h_gen_pdgid;
  v_h_trgen_mass = v_h_gen_mass;
  v_h_trgen_process = v_h_gen_process;
  tree_gen->Fill();
  if (return_flag) return;

  // Reconstructed taus
  edm::Handle<pat::TauCollection> taus;
  iEvent.getByToken(t_taus, taus);
  edm::Handle<pat::MuonCollection> muons;
  iEvent.getByToken(t_muons, muons);
  edm::Handle<pat::ElectronCollection> electrons;
  iEvent.getByToken(t_electrons, electrons);
  edm::Handle<reco::VertexCollection> vtxs;
  iEvent.getByToken(t_vtx, vtxs);
  edm::Handle<pat::JetCollection> jets;
  iEvent.getByToken(t_jets, jets);

  // Ensure that we have two reconstructed taus
  // int rec_channel = -1; // 3 - tt; 1 - et; 2 - mt

  if ((taus->size() < 1) || (taus->size() < 2 && expected_rec_channel > 2))
  {
    if (debug) std::cout << "less than expected reco taus: " << taus->size() << "\n";
    cutflow->Fill(2);
    return;
  }
  // // 3rd lepton veto
  // if (expected_rec_channel > 2)
  // {
  //   if (taus->size() < 2 || additional_ele(electrons[0]) || additional_mu(muons[0])) rec_channel = -1;
  // }
  // else if (expected_rec_channel == 2)
  // {
  //   if ((taus->size() < 1 && muons->size() < 1) || additional_ele(electrons[0]) || additional_mu(muons[1])) rec_channel = -1;
  // }
  // else if (expected_rec_channel == 1)
  // {
  //   if ((taus->size() < 1 && electrons->size() < 1) || additional_ele(electrons[1]) || additional_mu(muons[0])) rec_channel = -1;
  // }
  // else
  // {
  //   std::cout << "Skip since reco and gen channel do not match\n";
  //   cutflow->Fill(2);
  //   return;
  // }

  if (expected_rec_channel > 2) // tt channel
  {

    // Ensure that we can match both taus to different generator taus
    const float min_dr = 0.3; // Minimum deltaR valid for matching
    const auto idx1 = FindTau(taus, t1_vis_p4, min_dr);
    const auto idx2 = FindTau(taus, t2_vis_p4, min_dr);
    if (idx1 == -1 || idx2 == -1)
    {
      if (debug) std::cout << "one of the taus not matched to gen: idx1=" << idx1 << "; idx2="<< idx2 << " \n";
      cutflow->Fill(3);
      return;
    }
    if (idx1 == idx2)
    {
      if (debug) std::cout << "same matched taus: idx1=" << idx1 << " == idx2="<< idx2 << " \n";
      cutflow->Fill(3);
      return;
    }

    // Reco Jets njets
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>> l1, l2;
      l1.SetPxPyPzE(taus->at(idx1).p4().Px(), taus->at(idx1).p4().Py(), taus->at(idx1).p4().Pz(), taus->at(idx1).p4().E());
      l2.SetPxPyPzE(taus->at(idx2).p4().Px(), taus->at(idx2).p4().Py(), taus->at(idx2).p4().Pz(), taus->at(idx2).p4().E());
      njets = CountJets(jets, l1, 0.5, l2, 0.5);

    // Ensure that both reco taus have a reconstructed decay mode
    v_t1_rec_dm = taus->at(idx1).decayMode();
    v_t2_rec_dm = taus->at(idx2).decayMode();
    if (v_t1_rec_dm < 0 || v_t2_rec_dm < 0)
    {
      if (debug) std::cout << "Ensure that both taus have a reconstructed decay mode: v_t1_rec_dm=" << v_t1_rec_dm << " == v_t2_rec_dm="<< v_t2_rec_dm << " \n";
      cutflow->Fill(4);
      return;
    }

    // Ensure that the reco taus have opposite charge
    v_t1_rec_q = taus->at(idx1).charge();
    v_t2_rec_q = taus->at(idx2).charge();
    if (v_t1_rec_q == v_t2_rec_q)
    {
      if (debug) std::cout << "Ensure that the taus have opposite charge: v_t1_rec_q=" << v_t1_rec_q << " == v_t2_rec_q="<< v_t2_rec_q << " \n";
      cutflow->Fill(5);
      return;
    }
    if (v_t1_rec_q == 0 || v_t2_rec_q == 0)
    {
      if (debug) std::cout << "Ensure that the taus have charge: v_t1_rec_q=" << v_t1_rec_q << " == v_t2_rec_q="<< v_t2_rec_q << " \n";
      cutflow->Fill(5);
      return;
    }

    // Fill four-vector
    auto t1_rec_p4 = taus->at(idx1).p4();
    auto t2_rec_p4 = taus->at(idx2).p4();
    SetP4Values(t1_rec_p4, v_t1_rec);
    SetP4Values(t2_rec_p4, v_t2_rec);
    lv_tau1_rec.SetPxPyPzE(t1_rec_p4.Px(),t1_rec_p4.Py(),t1_rec_p4.Pz(),t1_rec_p4.E());
    lv_tau2_rec.SetPxPyPzE(t2_rec_p4.Px(),t2_rec_p4.Py(),t2_rec_p4.Pz(),t2_rec_p4.E());

    // Apply baseline selection
    if (baseline)
    {
      if (taus->at(idx1).pt() < 40) { cutflow->Fill(6); return; }
      if (taus->at(idx2).pt() < 40) { cutflow->Fill(6); return; }

      if (std::abs(taus->at(idx1).eta()) > 2.1) { cutflow->Fill(6); return; }
      if (std::abs(taus->at(idx2).eta()) > 2.1) { cutflow->Fill(6); return; }

      if (deltaR(t1_rec_p4, t2_rec_p4) < 0.5) { cutflow->Fill(6); return; }

      const auto nameDM = "decayModeFinding";
      if (taus->at(idx1).tauID(nameDM) < 0.5) { cutflow->Fill(6); return; }
      if (taus->at(idx2).tauID(nameDM) < 0.5) { cutflow->Fill(6); return; }

      const auto nameIso = "byVLooseIsolationMVArun2v1DBoldDMwLT";
      if (taus->at(idx1).tauID(nameIso) < 0.5) { cutflow->Fill(6); return; }
      if (taus->at(idx2).tauID(nameIso) < 0.5) { cutflow->Fill(6); return; }
    }
    tree_tt->Fill();
  }
  else if (expected_rec_channel == 2) // mt channel
  {
    // Matching
      const float min_dr = 0.3; // Minimum deltaR valid for matching
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>> t1_vis_p4_new;
      t1_vis_p4_new.SetPxPyPzE(t1_p4_vis.p4_vis.Px(), t1_p4_vis.p4_vis.Py(), t1_p4_vis.p4_vis.Pz(), t1_p4_vis.p4_vis.E());
      const auto idx1 = FindTau(muons, t1_vis_p4_new, min_dr); //TODO: check with def
      const auto idx2 = FindTau(taus, t2_vis_p4, min_dr);
      if (idx1 == -1 || idx2 == -1 || idx1 == idx2)
      {
        if (debug) std::cout << "Bad matching: idx1=" << idx1 << "; idx2="<< idx2 << " \n";
        cutflow->Fill(3);
        return;
      }

    // Reco Jets njets
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>> l1, l2;
      l1.SetPxPyPzE(muons->at(idx1).p4().Px(), muons->at(idx1).p4().Py(), muons->at(idx1).p4().Pz(), muons->at(idx1).p4().E());
      l2.SetPxPyPzE(taus->at(idx2).p4().Px(), taus->at(idx2).p4().Py(), taus->at(idx2).p4().Pz(), taus->at(idx2).p4().E());
      njets = CountJets(jets, l1, 0.5, l2, 0.5);

    // Opposite charge
      v_t1_rec_q = muons->at(idx1).charge();
      v_t2_rec_q = taus->at(idx2).charge();
      if ((v_t1_rec_q == v_t2_rec_q) || (v_t1_rec_q == 0 || v_t2_rec_q == 0))
      {
        if (debug) std::cout << "Bad charge of taus: v_t1_rec_q=" << v_t1_rec_q << " == v_t2_rec_q="<< v_t2_rec_q << " \n";
        cutflow->Fill(5);
        return;
      }

    // Fill reco four-vector
      auto t1_rec_p4 = muons->at(idx1).p4();
      auto t2_rec_p4 = taus->at(idx2).p4();
      SetP4Values(t1_rec_p4, v_t1_rec);
      SetP4Values(t2_rec_p4, v_t2_rec);
      lv_tau1_rec.SetPxPyPzE(t1_rec_p4.Px(),t1_rec_p4.Py(),t1_rec_p4.Pz(),t1_rec_p4.E());
      lv_tau2_rec.SetPxPyPzE(t2_rec_p4.Px(),t2_rec_p4.Py(),t2_rec_p4.Pz(),t2_rec_p4.E());

    // Apply baseline selection
    if (baseline)
    {
      // muon
        if (muons->at(idx1).pt() < 20) { cutflow->Fill(6); return; }
        if (std::abs(muons->at(idx1).eta()) > 2.1){ cutflow->Fill(6); return; }
        if (fabs(muons->at(idx1).muonBestTrack()->dxy(vtxs->at(0).position())) < 0.045 && fabs(muons->at(idx1).muonBestTrack()->dz(vtxs->at(0).position()))  < 0.2) { cutflow->Fill(6); return; }
      // tau
        v_t2_rec_dm = taus->at(idx2).decayMode();
        const auto nameDM = "decayModeFinding";

        if (taus->at(idx2).pt() < 20) { cutflow->Fill(6); return; }
        if (std::abs(taus->at(idx2).eta()) > 2.1) { cutflow->Fill(6); return; } // TODO: should be 2.3
        if (v_t2_rec_dm < 0) { cutflow->Fill(4); return; }
        if (taus->at(idx2).tauID(nameDM) < 0.5) { cutflow->Fill(6); return; }
        // TODO: fabs(packedLeadTauCand->dz()) < 0.2  # The PackedCandidate::dz() method is wrt. the first PV by default
        // TODO: check
        const auto nameIso = "byVLooseIsolationMVArun2v1DBoldDMwLT";
        if (taus->at(idx2).tauID(nameIso) < 0.5) { cutflow->Fill(6); return;}

      // Pair
        if (deltaR(t1_rec_p4, t2_rec_p4) < 0.5) { cutflow->Fill(6); return; }
      // TODO: Post-sync ntuple ?
    }
    tree_mt->Fill();
  }
  else if (expected_rec_channel == 1) // et channel
  {
    // Matching
      const float min_dr = 0.3; // Minimum deltaR valid for matching
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>> t1_vis_p4_new;
      t1_vis_p4_new.SetPxPyPzE(t1_p4_vis.p4_vis.Px(), t1_p4_vis.p4_vis.Py(), t1_p4_vis.p4_vis.Pz(), t1_p4_vis.p4_vis.E());
      const auto idx1 = FindTau(electrons, t1_vis_p4_new, min_dr); //TODO: check with def
      const auto idx2 = FindTau(taus, t2_vis_p4, min_dr);
      if ((idx1 == -1 || idx2 == -1) || (idx1 == idx2))
      {
        if (debug) std::cout << "Bad matching: idx1=" << idx1 << "; idx2="<< idx2 << " \n";
        cutflow->Fill(3);
        return;
      }

    // Reco Jets njets
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>> l1, l2;
      l1.SetPxPyPzE(electrons->at(idx1).p4().Px(), electrons->at(idx1).p4().Py(), electrons->at(idx1).p4().Pz(), electrons->at(idx1).p4().E());
      l2.SetPxPyPzE(taus->at(idx2).p4().Px(), taus->at(idx2).p4().Py(), taus->at(idx2).p4().Pz(), taus->at(idx2).p4().E());
      njets = CountJets(jets, l1, 0.5, l2, 0.5);

    // Opposite charge
      v_t1_rec_q = electrons->at(idx1).charge();
      v_t2_rec_q = taus->at(idx2).charge();
      if ((v_t1_rec_q == v_t2_rec_q) || (v_t1_rec_q == 0 || v_t2_rec_q == 0))
      {
        if (debug) std::cout << "Bad charge of taus: v_t1_rec_q=" << v_t1_rec_q << " == v_t2_rec_q="<< v_t2_rec_q << " \n";
        cutflow->Fill(5);
        return;
      }

    // Fill reco four-vector
      auto t1_rec_p4 = electrons->at(idx1).p4();
      auto t2_rec_p4 = taus->at(idx2).p4();
      SetP4Values(t1_rec_p4, v_t1_rec);
      SetP4Values(t2_rec_p4, v_t2_rec);
      lv_tau1_rec.SetPxPyPzE(t1_rec_p4.Px(),t1_rec_p4.Py(),t1_rec_p4.Pz(),t1_rec_p4.E());
      lv_tau2_rec.SetPxPyPzE(t2_rec_p4.Px(),t2_rec_p4.Py(),t2_rec_p4.Pz(),t2_rec_p4.E());

    // Apply baseline selection
    if (baseline)
    {
      // electron
        if (electrons->at(idx1).pt() < 25) { cutflow->Fill(6); return; }
        if (std::abs(electrons->at(idx1).eta()) > 2.1){ cutflow->Fill(6); return; }
        if (fabs(electrons->at(idx1).gsfTrack()->dxy(vtxs->at(0).position())) < 0.045 &&
            fabs(electrons->at(idx1).gsfTrack()->dz(vtxs->at(0).position())) < 0.2) { cutflow->Fill(6); return; }
        // TODO:
          // elec.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS))) <=1     and
          // elec.passConversionVeto()
      // tau
        v_t2_rec_dm = taus->at(idx2).decayMode();
        const auto nameDM = "decayModeFinding";

        if (taus->at(idx2).pt() < 20) { cutflow->Fill(6); return; }
        if (std::abs(taus->at(idx2).eta()) > 2.1) { cutflow->Fill(6); return; } // TODO: should be 2.3
        if (v_t2_rec_dm < 0) { cutflow->Fill(4); return; }
        if (taus->at(idx2).tauID(nameDM) < 0.5) { cutflow->Fill(6); return; }
        // TODO: fabs(packedLeadTauCand->dz()) < 0.2  # The PackedCandidate::dz() method is wrt. the first PV by default
        // TODO: check if this should be removed
        const auto nameIso = "byVLooseIsolationMVArun2v1DBoldDMwLT";
        if (taus->at(idx2).tauID(nameIso) < 0.5) { cutflow->Fill(6); return;}

      // Pair
        if (deltaR(t1_rec_p4, t2_rec_p4) < 0.5) { cutflow->Fill(6); return; }
      // TODO: Post-sync ntuple ?
    }
    tree_et->Fill();
  }
  else return;

  tree->Fill();
}

void ntupleBuilder::beginJob() {}

void ntupleBuilder::endJob() {}

DEFINE_FWK_MODULE(ntupleBuilder);
