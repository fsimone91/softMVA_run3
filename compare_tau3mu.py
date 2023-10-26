import ROOT
from ROOT import RDataFrame

import numpy as np
import pickle

from utils import *
from data_preparation import *
 
# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()

# Batch mode
ROOT.gROOT.SetBatch(ROOT.kTRUE)


def call_th1_mod(df1, df2, f1, f2):
       print("plotting feature ",f)
       #binning from dictionary
       x1 = feature_dict[f1][0]
       x2 = feature_dict[f1][1]
       h1 = df1.Histo1D((f1, f1, 50, x1, x2), f1)
       h2 = df2.Histo1D((f1, f1, 50, x1, x2), f2)
       return h1, h2

if __name__ == "__main__":

   path_run3_sig = "/eos/user/f/fsimone/softMVA_run3/fold_0_signal_10M.root"
   path_run3_bkg = "/eos/user/f/fsimone/softMVA_run3/fold_0_background_10M.root"

   path_t23m_sig = "/eos/user/f/fsimone/softMVA_run3/AnalysedTree_MC_2018Ds_tau3mu_14dec.root"
   path_t23m_bkg = "/eos/user/f/fsimone/softMVA_run3/AnalysedTree_data_2018D_tau3mu_14dec.root"

   df_run3_sig = ROOT.RDataFrame("Events", path_run3_sig) 
   df_run3_bkg = ROOT.RDataFrame("Events", path_run3_bkg)            
   df_t23m_sig = ROOT.RDataFrame("TreeMu3",path_t23m_sig) 
   df_t23m_bkg = ROOT.RDataFrame("TreeMu3",path_t23m_bkg) 

   #plot interesting features
   c = ROOT.TCanvas("c", "c", 600, 800)
   histo_pairs_sig = []
   histo_pairs_bkg = []
   for f in features:
       histo_pairs_sig.append(call_th1_mod(df_run3_sig, df_t23m_sig, f, feature_dict_tau3mu[f]))
       histo_pairs_bkg.append(call_th1_mod(df_run3_bkg, df_t23m_bkg, f, feature_dict_tau3mu[f]))
   for index, histo_pair in enumerate(histo_pairs_sig):
       draw_th1(histo_pair[0], histo_pair[1], features[index], "run3_sig", "t3mu_MC", c, "compare_sig")
   for index, histo_pair in enumerate(histo_pairs_bkg):
       draw_th1(histo_pair[0], histo_pair[1], features[index], "run3_bkg", "t3mu_data", c, "compare_bkg")
