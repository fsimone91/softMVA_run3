import sys, os, subprocess, json
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score

from pprint import pprint

import ROOT
from ROOT import RDataFrame

import matplotlib as mpl
# https://matplotlib.org/faq/usage_faq.html
mpl.use('Agg')
import matplotlib.pyplot as plt
from math import cos

base_dir = '/afs/cern.ch/work/f/fsimone/softMVA_run3'
bdt_dir = '/afs/cern.ch/user/w/wangz/public/softmuon/'

def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

##feature_names = [
##    ### Good
##    "trkValidFrac",
##    "glbTrackProbability",
##    "nLostHitsInner",
##    "nLostHitsOuter",
##    "trkKink",
##    "chi2LocalPosition",
##    "match2_dX",
##    "match2_pullX",
##    "match1_dX",
##    "match1_pullX",
##
##    ### Weak but useful
##    "nPixels",
##    "nValidHits",
##    "nLostHitsOn",
##    "match2_dY",
##    "match1_dY",
##    "eta",
##    "match2_pullY",
##    "match1_pullY",
##    "match2_pullDyDz",
##    "match1_pullDyDz",
##    "match2_pullDxDz",
##    "match1_pullDxDz",
##    "pt"
##]

#keys: names in xgboost training
#values: names in t3mu ntuples (1/2/3 missing depending on muon)
feature_dict = {
   'trkValidFrac' :  'trkValidFrac',
   'glbTrackProbability': 'cQ_gTP_',
   'nLostHitsInner': 'nLostHitsInner',
   'nLostHitsOuter': 'nLostHitsOuter',
   'trkKink': 'cQ_tK_',
   'chi2LocalPosition':'cQ_Chi2LP_',
   'match2_dX':'match2_dX_',
   'match2_pullX':'match2_pullX_',
   'match1_dX':'match1_dX_',
   'match1_pullX':'match1_pullX_',

   'nPixels':'nValidPixelHits',
   'nValidHits':'nValidTrackerHits',
   'nLostHitsOn':'nLostHitsOn',
   'match2_dY':'match2_dY_',
   'match1_dY':'match1_dY_',
   'eta':'Etamu',
   'match2_pullY':'match2_pullY_',
   'match1_pullY':'match1_pullY_',
   'match2_pullDyDz':'match2_pullDyDz_',
   'match1_pullDyDz':'match1_pullDyDz_',
   'match2_pullDxDz':'match2_pullDxDz_',
   'match1_pullDxDz':'match1_pullDxDz_',
   'pt':'Ptmu'
}

#model = load_pkl(base_dir+"/MIT_training/Run2022-20230930-1539-Event0.pkl")
model = xgb.Booster()
model.load_model(bdt_dir+'/weight_muon_mva_pt2_pkl/Run2022-20231023-1746-Event0.model')
#workaround to get original feature names https://stackoverflow.com/questions/76664025/xgboost-save-model-loses-feature-names-when-saving
features = json.load(open(bdt_dir+'/weight_muon_mva_pt2_pkl/Run2022-20231023-1746-Event0.features'))
model.feature_names = features
print(model)
print(model.feature_names)

def get_input_features(df, train_list, cuts=''):
    if cuts=='': return df[train_list].to_numpy()
    _df = df[df.eval(cuts)]
    return _df[train_list].to_numpy()

def get_arrays(tree, branch_list):
    _dict = {}
    for _br in branch_list:
        _dict[_br] = getattr(tree[_br].arrays(), _br)
    return pd.DataFrame.from_dict(_dict)

inputfiles = {
   'data': base_dir+'/t3mu_2022/AnalysedTree_data_2022F_tau3mu_merged_v2.root',
   'mc' : base_dir+'/t3mu_2022/AnalysedTree_MC_Ds_postE_tau3mu_merged_v2.root'
}

def prepare_dataset(inputfilename, muonlabel):
   #modify dictionary adding muonlabel and invert
   d = dict((feature_dict[k]+muonlabel, k) for k in feature_dict)

   # load data
   input_file = uproot.open(inputfilename)
   input_tree = get_arrays(input_file['FinalTree'], list(d.keys()))

   #rename columns to match training
   input_tree = input_tree.rename(columns=d, errors="raise")

   return input_tree

def evaluate_model(input_tree, model, preselection, branch_name):
    input_X = get_input_features(input_tree, model.feature_names, preselection)
    dtest = xgb.DMatrix(input_X, feature_names=model.feature_names)

    print(input_X)
    #evaluate model
    score = model.predict(dtest)

    input_tree[branch_name] = score

    return input_tree

def main():
    muon_labels = ['1', '2', '3']
    input_labels = ['data', 'mc']

    #features for final ntuples
    branch_list = [(feature_dict[k]+m) for k in feature_dict for m in muon_labels]
    branch_list = branch_list + ['MVA1', 'MVA2', 'MVA3', 'MVASoft1', 'MVASoft2', 'MVASoft3', 'dimu_OS1', 'dimu_OS2', 'tripletMass']
    print(branch_list)
    for i in input_labels:
       scores = []
       #open original file
       if_total = uproot.open(inputfiles[i])
       df_total = get_arrays(if_total['FinalTree'], branch_list)
       for m in muon_labels:
          df = prepare_dataset(inputfiles[i], m)
          df = evaluate_model(df,model,'', 'softmva_run3_mu'+m)
          score = df['softmva_run3_mu'+m]
          df_total['softmva_run3_mu'+m] = score
 
       # Write in a TTree
       fileName = "t3mminitree_"+i;
       df_total.to_csv(fileName+".csv")
       rdf = ROOT.RDF.FromCSV(fileName+".csv")
       rdf.Snapshot('mytree'+i, fileName+'.root')

if __name__ == "__main__":
    main()



