import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils

# to run:
# cmsRun workspace/ntupleBuilder/python/run_cfi.py ; tput bell

process = cms.Process("ntupleBuilder")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = "WARNING"
process.MessageLogger.categories.append("ntupleBuilder")
process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit=cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(True))

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.register('loginterval', 1000, mytype=VarParsing.varType.int)
options.parseArguments()
# print options.inputFiles;

# Set the maximum number of events to be processed (-1 processes all events)
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))

# Define files of dataset
# https://cmsweb.cern.ch/das/request?view=plain&limit=50&instance=prod%2Fglobal&input=file+dataset%3D%2FDYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8%2FRunIIFall17MiniAODv2-PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1%2FMINIAODSIM
files=[
    '/store/mc/RunIIFall17MiniAODv2/SUSYGluGluToHToTauTau_M-800_TuneCP5_13TeV-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/8EF94B44-8E42-E811-801C-002590FD5A3A.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/50000/F22BEB20-6B44-E811-8D75-0CC47A7C351E.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/0C0443C1-4944-E811-A399-0CC47AD98BC6.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/0AAB8FD4-8A44-E811-BA06-0025905B85BC.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/0A6C2160-9844-E811-BD91-0CC47A4D7668.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/0A66C6BD-8244-E811-B80B-0025905B8572.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/0A0E10BD-8C44-E811-A684-0CC47A4C8E26.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/06D236B3-8244-E811-9CD4-0CC47A7C3472.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/04F385BD-4644-E811-9C1B-0CC47A6C138A.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/04AAA539-7B44-E811-BC75-0CC47A4C8EA8.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/043D80A2-7C44-E811-826F-0CC47A7C3410.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/00F3F4F6-4244-E811-BCB4-90B11C2AA430.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/0078FEA2-8044-E811-A405-0025905A6126.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/20000/0056F0F4-4F44-E811-B415-FA163E5B5253.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/F2283B5C-6044-E811-B61D-0025905B859A.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/F034EF3E-4244-E811-B9D5-FA163EFB71BE.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/EE730596-5B44-E811-A721-0025905A613C.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/E0BC5502-4644-E811-901E-FA163EE67077.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/DED514E2-7644-E811-9F36-0CC47A7C34A0.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/DA1BE8CB-5444-E811-BDBF-0025905A48FC.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/D86C0EA4-5744-E811-9543-0CC47A4D7604.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/CC07F3E3-5344-E811-A090-FA163E6ED8A3.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/BC194ABA-5644-E811-884C-0CC47A78A2F6.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/B81792DC-4244-E811-8C5A-02163E01A0D0.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/B49537A2-4E44-E811-BAD7-FA163E8686A0.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/B27AAEBA-5944-E811-884C-0CC47A4C8E56.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/B0950FD6-6944-E811-9FF3-0CC47A4D7670.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/AA7E6CE9-CA44-E811-9BD5-0CC47A4D7606.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/A419B948-CB44-E811-95EC-00259048AE50.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/9EC6E114-4444-E811-AEE8-FA163EE0A861.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/9803F4B2-5C44-E811-B43D-0CC47A78A30E.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/8EF3BB39-5F44-E811-8333-0CC47A4C8EC8.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/84D2F40E-4044-E811-8FA6-90B11CBCFFD0.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/80FF66DC-4A44-E811-80DC-FA163EFF0F8E.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/7CF15953-6144-E811-89E6-0CC47A4D762E.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/7CBC238D-5B44-E811-8C02-0CC47A4C8E98.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/78771AC4-6244-E811-805E-0025905B856E.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/705503B4-6844-E811-AFB1-0CC47A4D7698.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/701668BB-4F44-E811-9FDD-02163E01A169.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/700E55B2-5F44-E811-849D-0CC47A4D76C6.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/6C261642-4644-E811-BA2E-FA163E28C9AF.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/6A37971B-4B44-E811-9B02-02163E01A169.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/664764D5-6744-E811-987F-0CC47A7C3444.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/62828019-CB44-E811-A7FF-7CD30AD0A78C.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/60A78AA5-5844-E811-87F9-0025905A608C.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/5EF33F30-6544-E811-8E67-0CC47A4D768C.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/5624214D-5C44-E811-B2FC-0025905B8610.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/54B4F9C2-3E44-E811-998F-002481CFB40E.root',
    # '/store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/70000/4AFB8A91-4C44-E811-B34D-FA163EA35F18.root',
    #
    # /GluGluHToTauTau_M125_13TeV_powheg_pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/MINIAODSIM
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/F8E1F88F-3022-E911-AEF0-44A842CFCA1A.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/EE5A0F69-0521-E911-86C3-44A842CFD626.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/EE054650-5A21-E911-9646-484D7E8DF0D3.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/ECF531F6-3E21-E911-AA18-44A842CF05D9.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/E6C9C69A-4B21-E911-A92C-B499BAAC0658.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/DCCA61BA-2B21-E911-94C7-44A842CFD667.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/D695C2E9-6221-E911-8B8B-001CC47BEE5E.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/D4018420-1A21-E911-9C8B-6C3BE5B59210.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/D24C64BB-2321-E911-949A-6C3BE5B5A038.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/D0BA83B1-3D21-E911-A905-484D7E8DF051.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/C46A701F-EF20-E911-A144-44A842CFD5F2.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/BA70ABEA-4121-E911-8AF6-484D7E8DF051.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/A0BAD3A1-2321-E911-AABD-B499BAAC04E6.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/A0554F70-5721-E911-A9D8-44A842CFD5D8.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/A03AC4B7-3D21-E911-A4E0-44A842CFD64D.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/98A92539-4E21-E911-9A17-44A842CFD60C.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/94429AEE-3E21-E911-AAE9-44A842CFC97E.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/8CE664B6-3D21-E911-9CF0-484D7E8DF0FA.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/82DCC89F-1321-E911-83E9-44A842CFD64D.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/7E2573F9-5B21-E911-A92B-44A842CFD5D8.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/7AF95723-4621-E911-86DF-484D7E8DF051.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/763A7921-EF20-E911-BD8B-44A842CF05B2.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/70B2A1A3-0E21-E911-B957-44A842CFD5CB.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/6CBDE5C4-4721-E911-B941-44A842CFC9F3.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/60B64277-0521-E911-9706-B499BAAC0A22.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/524C7FB6-3D21-E911-A54D-44A842CF05BF.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/46CE51F5-1A21-E911-BC47-44A842CFD64D.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/406470FA-3E21-E911-9F96-001CC4A61C52.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/3C01C018-2921-E911-B992-B499BAAC04E6.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/280C50A7-0E21-E911-A214-484D7E8DF085.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/20301EA9-0E21-E911-93F9-44A842CFD64D.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/1292CBA1-5021-E911-B3DB-B499BAAC0658.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/0A52B0C2-3D21-E911-BC74-D8D385B0EE2E.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/0604D195-1421-E911-A3BD-6C3BE5B59210.root',
    # '/store/mc/RunIIFall17MiniAODv2/GluGluHToTauTau_M125_13TeV_powheg_pythia8/MINIAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/10000/047009D4-3D21-E911-9727-6C3BE5B541F8.root',
    # "file:miniAOD-prod_PAT.root",
    # root://dcache-cms-xrootd.desy.de:1094/
    # root://cms-xrd-global.cern.ch/
    # root://cms-xrd-global.cern.ch//store/mc/RunIIFall17MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/50000/F22BEB20-6B44-E811-8D75-0CC47A7C351E.root
]
# files = ['file:/eos/user/o/ohlushch/Nostradamas/mass_regression/CMSSW_9_4_13_UL1/DYtest_Run2_2017_FastSim_auto_phase1_2017_realistic_DYtest_1000_wm/91GeV/0/miniAOD-prod_PAT.root'] # ['file:GGHminiAOD-prod_PAT.root']  # files

# this is a hook for jdl! don't remove
#__filelist
if len(options.inputFiles) > 0: filelist = options.inputFiles
tmpOutputFile = "ntuple.root"
if len(options.outputFile) > 0: tmpOutputFile = options.outputFile
process.source = cms.Source(
    "PoolSource", fileNames=cms.untracked.vstring(*files))

# Set global tag
# process.GlobalTag.globaltag = "START53_V27::All"

# Number of events to be skipped (0 by default)
process.source.skipEvents = cms.untracked.uint32(0)

settype = "DYtest"  # "GGH"  # "GGH"  # "DY"
if '/DYJets' in files[0]: settype = "DYtest"
elif any(word in files[0] for word in ['/GluGluHToTauTau','/GGH', '/SUSYGluGluToHToTauTau']): settype = "GGHtest"
elif any(word in files[0] for word in ['/VBFHToTauTau','/QQH', '/SUSYGluGluToBBHToTauTau']): settype = "QQHHtest"

#__settype
if "/store/mc" in files[0]:
    settype += "_central"
# Register fileservice for output file
print settype
process.ntupleBuilder = cms.EDAnalyzer("ntupleBuilder",
    processType=cms.string(settype),
    debug=cms.bool(False),
    baseline=cms.bool(True),
)
process.TFileService = cms.Service(
    "TFileService", fileName=cms.string(tmpOutputFile))

process.p = cms.Path(process.ntupleBuilder)
