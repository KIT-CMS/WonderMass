from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'crabdir_SUSYM800_2017'
config.General.workArea = 'Official17'
config.General.transferLogs = True

config.section_("User")
config.User.voGroup = 'dcms'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'ntupleBuilder/python/run_test_cfi.py'

config.section_("Data")
config.Data.inputDataset = '/SUSYGluGluToHToTauTau_M-800_TuneCP5_13TeV-pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.totalUnits = -1
config.Data.outLFNDirBase = '/store/user/ohlushch/MassRegression/Official17/'
config.Data.publication = False
config.Data.allowNonValidInputDataset = True

config.section_("Site")
config.Site.storageSite = 'T2_DE_DESY'
