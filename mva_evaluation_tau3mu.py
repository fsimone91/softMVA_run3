import ROOT

TMVA = ROOT.TMVA
TFile = ROOT.TFile

# Enable multi-threading
#ROOT.ROOT.EnableImplicitMT()

import numpy as np
import pickle
from array import array

from utils import *

## NOT WORKING ON TTREE WITH /D BRANCHES
def eval_model(input_file_path, tree_name, weight_path):
   print(input_file_path)
   ROOT.gInterpreter.ProcessLine('''
            TMVA::Experimental::RReader model("{}");
            '''.format(weight_path))

   # variables used for training
   n_var = len(ROOT.model.GetVariableNames())
   ROOT.gInterpreter.ProcessLine('''
            compute = TMVA::Experimental::Compute<{}, float>(model);
            '''.format(n_var))

   # importing input tree as df
   df = ROOT.RDataFrame(tree_name, input_file_path)
   # renaming branches and cast to float
   for f in ROOT.model.GetVariableNames():
       df = df.Define(f, feature_dict_tau3mu[f])

   #evaluating the model
   df = df.Define("mu_softMVArun3", ROOT.compute, ROOT.model.GetVariableNames())

   h = df.Histo1D("mu_softMVArun3")
   h.Draw()
   
   return df
   

def add_eval_branch(in_fname='infile.root'):

   ## feature_arrays = []
   ## spectat_arrays = []
   ## empty_array = array('f', [-1.0])
   ## for index, f in enumerate(features):
   ##    feature_arrays.append(empty_array)
   ##    #feature_arrays[index] = array('f', [-1.0]);
   ## for index, s in enumerate(spectators):
   ##    spectat_arrays.append(empty_array)
   ##    #spectat_arrays[index] = array('f', [-1.0]);

   a_pt = array('f', [-1.0])
   a_eta = array('f', [-1.0])

   a_trkLayers = array('f', [-1.0])
   a_nPixels = array('f', [-1.0])
   a_trkValidFrac = array('f', [-1.0])
   a_nValidHits = array('f', [-1.0])

   a_staNormChi2 = array('f', [-1.0])
   a_trkNormChi2 = array('f', [-1.0])
   a_glbNormChi2 = array('f', [-1.0])
   a_staRelChi2 = array('f', [-1.0])
   a_trkRelChi2 = array('f', [-1.0])

   a_chi2LocalMomentum = array('f', [-1.0])
   a_chi2LocalPosition = array('f', [-1.0])
   a_nStations = array('f', [-1.0])
   a_trkKink = array('f', [-1.0])
   a_segmentCom = array('f', [-1.0])

   a_weight = array('f', [-1.0])
   a_evt = array('f', [-1.0])
   a_sim_pdgId = array('f', [-1.0])
   a_sim_mpdgId = array('f', [-1.0])
   a_sim_type = array('f', [-1.0])

   reader_softMVA = TMVA.Reader("!Color:!Silent");

   #for index, f in enumerate(features):
   #   reader_softMVA.AddVariable(feature_expression[f], feature_arrays[index])
   #for index, s in enumerate(spectators):
   #   reader_softMVA.AddSpectator(s, spectat_arrays[index])

   reader_softMVA.AddVariable("pt" ,a_pt)
   reader_softMVA.AddVariable("eta",a_eta)

   reader_softMVA.AddVariable("trkLayers" ,a_trkLayers) 
   reader_softMVA.AddVariable("nPixels" ,a_nPixels)
   reader_softMVA.AddVariable("trkValidFrac" ,a_trkValidFrac)
   reader_softMVA.AddVariable("nValidHits" ,a_nValidHits)

   reader_softMVA.AddVariable("staNormChi2>50?50:staNormChi2" ,a_staNormChi2)
   reader_softMVA.AddVariable("trkNormChi2>50?50:trkNormChi2" ,a_trkNormChi2)
   reader_softMVA.AddVariable("glbNormChi2>50?50:glbNormChi2" ,a_glbNormChi2)
   reader_softMVA.AddVariable("staRelChi2>50?50:staRelChi2" ,a_staRelChi2 )
   reader_softMVA.AddVariable("trkRelChi2>50?50:trkRelChi2" ,a_trkRelChi2 )

   reader_softMVA.AddVariable("chi2LocalMomentum>150?150:chi2LocalMomentum" ,a_chi2LocalMomentum)
   reader_softMVA.AddVariable("chi2LocalPosition>50?50:chi2LocalPosition" ,a_chi2LocalPosition)
   reader_softMVA.AddVariable("nStations" ,a_nStations)
   reader_softMVA.AddVariable("log(1+trkKink)<3?3:log(1+trkKink)" ,a_trkKink)
   reader_softMVA.AddVariable("segmentComp" ,a_segmentCom) 

   reader_softMVA.AddSpectator("weight" ,a_weight)
   reader_softMVA.AddSpectator("evt" ,a_evt)
   reader_softMVA.AddSpectator("sim_pdgId" ,a_sim_pdgId)
   reader_softMVA.AddSpectator("sim_mpdgId" ,a_sim_mpdgId)
   reader_softMVA.AddSpectator("sim_type" ,a_sim_type)
   
   reader_softMVA.BookMVA( "BDT", "dataset_26aug23/weights/TMVA_BDT_Classification_BDT.weights.xml" );

   print('opening file ',in_fname)
   file_ = ROOT.TFile(in_fname, "READ")
   old_tree = [ file_.Get('TreeMu1'), file_.Get('TreeMu2'), file_.Get('TreeMu3') ]
   out_fname = in_fname.split(".root")[0] + '_26aug23_softMVA.root'
   new_file = ROOT.TFile(out_fname,"RECREATE")
  
   new_tree = [file_.Get('TreeMu1').CloneTree(0), file_.Get('TreeMu2').CloneTree(0), file_.Get('TreeMu3').CloneTree(0)]
   #new_tree = [ ROOT.TTree('softMVA_Mu1', 'softMVA_Mu1'), ROOT.TTree('softMVA_Mu2', 'softMVA_Mu2'), ROOT.TTree('softMVA_Mu3', 'softMVA_Mu3') ]
 
   softMVA = [ array('f', [-99.]), array('f', [-99.]), array('f', [-99.]) ]
  
   for i in range(3): 
       new_tree[i].Branch("mu_softMVA_run3", softMVA[i], "mu_softMVA_run3/F")
   
   for i in range(3):
       nentries = old_tree[i].GetEntriesFast()
       print( old_tree[i].GetName() )
       for j in range(nentries):
          if (j%1000==0): print( "Processing ",j,"/",nentries," ...")
          old_tree[i].GetEntry(j)
          #print('muon pt, eta ',old_tree[i].mu_pt, old_tree[i].mu_eta )
          muon_score = -99.0
          a_pt[0]  = old_tree[i].mu_pt
          a_eta[0] = old_tree[i].mu_eta
                       
          a_trkLayers[0]    = old_tree[i].mu_trackerLayersWithMeasurement
          a_nPixels[0]      = old_tree[i].mu_Numberofvalidpixelhits
          a_trkValidFrac[0] = old_tree[i].mu_innerTrack_validFraction
          a_nValidHits[0]   = old_tree[i].mu_GLhitPattern_numberOfValidMuonHits
                       
          a_staNormChi2[0] = old_tree[i].mu_outerTrack_normalizedChi2
          a_trkNormChi2[0] = old_tree[i].mu_innerTrack_normalizedChi2
          a_glbNormChi2[0] = old_tree[i].mu_GLnormChi2
          a_staRelChi2[0]  = old_tree[i].mu_combinedQuality_trkRelChi2
          a_trkRelChi2[0]  = old_tree[i].mu_combinedQuality_staRelChi2
                       
          a_chi2LocalMomentum[0] = old_tree[i].mu_combinedQuality_chi2LocalMomentum
          a_chi2LocalPosition[0] = old_tree[i].mu_combinedQuality_chi2LocalPosition
          a_nStations[0]  = old_tree[i].mu_numberOfMatchedStations
          a_trkKink[0]    = old_tree[i].mu_combinedQuality_trkKink
          a_segmentCom[0] = old_tree[i].mu_segmentCompatibility

          muon_score = reader_softMVA.EvaluateMVA("BDT")
       
          softMVA[i][0] = muon_score
          #print('mva score ',muon_score)
          new_tree[i].Fill()
       
   new_file.cd()
   new_tree[0].Write()
   new_tree[1].Write()
   new_tree[2].Write()
   new_file.Close()
   file_.Close()

def main():
   add_eval_branch("/eos/user/f/fsimone/softMVA_run3/AnalysedTree_data_2018D_tau3mu_14dec.root")
   add_eval_branch("/eos/user/f/fsimone/softMVA_run3/AnalysedTree_MC_2018Ds_tau3mu_14dec.root")
   #df = eval_model("/eos/user/f/fsimone/softMVA_run3/AnalysedTree_data_2018D_tau3mu_14dec.root", "TreeMu3", "./dataset_nospectators/weights/TMVA_BDT_Classification_BDT.weights.xml")

if __name__=='__main__':
    main()
