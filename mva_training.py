import ROOT

TMVA = ROOT.TMVA
TFile = ROOT.TFile

# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()

import numpy as np
import pickle


from utils import *


# Set here preferred MVA
useXgboost = False
useTMVABDT = True

hasGPU = ROOT.gSystem.GetFromPipe("root-config --has-tmva-gpu") == "yes"
hasCPU = ROOT.gSystem.GetFromPipe("root-config --has-tmva-cpu") == "yes"

if __name__ == "__main__":

   if useTMVABDT:

      TMVA.Tools.Instance()

      outputFile = TFile.Open(output_folder+"/TMVA_BDT_ClassificationOutput.root", "RECREATE")

 
      factory = TMVA.Factory(
          "TMVA_BDT_Classification",
          outputFile,
          V=False,
          ROC=True,
          Silent=False,
          Color=True,
          AnalysisType="Classification",
          Transformations=None,
          Correlations=False,
      )
 
      loader = TMVA.DataLoader("dataset_26aug23")
      input_file_sig = TFile.Open(output_folder+"/fold_0_signal_jpsi_10M.root")
      input_file_bkg = TFile.Open(output_folder+"/fold_0_background_10M.root")

      # --- Register the training and test trees
      tree_sig = input_file_sig.Get("Events")
      tree_bkg  = input_file_bkg.Get("Events")
 
      n_sig = tree_sig.GetEntries()
      n_bkg = tree_bkg.GetEntries()
 
      # global event weights per tree
      w_sig = 1.0 #n_bkg / (n_sig+n_bkg)
      w_bkg = 1.0 #n_sig / (n_sig+n_bkg)
 
      loader.AddSignalTree(tree_sig, w_sig)
      loader.AddBackgroundTree(tree_bkg, w_bkg)

      # per-event weight
      loader.SetBackgroundWeightExpression("weight");

      # Apply additional cuts on the signal and background samples (can be different)
      mycuts = ""
      mycutb = ""
      
      # Input features
      for f in features:
         loader.AddVariable(feature_expression[f], f, "", 'F')

      # NOTE: TMVA::Experimental::RReader does not support spectator variables
      for s in spectators:
         loader.AddSpectator(s)
 
      # Tell the factory how to use the training and testing events
      # If no numbers of events are given, half of the events in the tree are used
      # for training, and the other half for testing:
      #    loader.PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
      # It is possible also to specify the number of training and testing events,
      # note we disable the computation of the correlation matrix of the input variables
       
      n_train_sig = 0.8 * n_sig
      n_train_bkg = 0.8 * n_bkg
       
      # build the string options for DataLoader::PrepareTrainingAndTestTree
       
      loader.PrepareTrainingAndTestTree(
          mycuts,
          mycutb,
          nTrain_Signal=n_train_sig,
          nTrain_Background=n_train_bkg,
          SplitMode="Random",
          SplitSeed=100,
          NormMode="NumEvents",
          V=False,
          CalcCorrelations=False,
      )

      # Book Method: Boosted Decision Trees
      factory.BookMethod(
         loader,
         TMVA.Types.kBDT,
         "BDT",
         V=False,
         NTrees=1000,
         MinNodeSize="2.5%",
         MaxDepth=6,
         BoostType="RealAdaBoost",
         AdaBoostBeta=0.3,
         UseBaggedBoost=True,
         BaggedSampleFraction=0.1,
         SeparationType="GiniIndex",
         nCuts=100,
      )
  
      # Train Methods
      factory.TrainAllMethods()
       
      # Test and Evaluate Methods
      factory.TestAllMethods()
       
      factory.EvaluateAllMethods()
       
      # Plot ROC Curve
      c1 = factory.GetROCCurve(loader)
      c1.Draw()
       
      # close outputfile to save output file
      outputFile.Close()
