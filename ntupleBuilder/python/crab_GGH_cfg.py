from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'crabdir_GluGluHToTauTau_M125_13TeV_powheg_pythia8'
config.General.workArea = 'Official17'
config.General.transferLogs = True

config.section_("User")
config.User.voGroup = 'dcms'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'workspace/ntupleBuilder/python/run_cfi.py'

config.section_("Data")
config.Data.inputDataset = '/GluGluHToTauTau_M125_13TeV_powheg_pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/MINIAODSIM'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.totalUnits = -1
config.Data.outLFNDirBase = '/store/user/ohlushch/MassRegression/Official17/'
config.Data.publication = False
config.Data.allowNonValidInputDataset = True

config.section_("Site")
config.Site.storageSite = 'T2_DE_DESY'
