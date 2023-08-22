import ROOT

TMVA = ROOT.TMVA
TFile = ROOT.TFile

# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()

import numpy as np
import pickle

from xgboost import XGBClassifier

from utils import *

def load_data(signal_filename, background_filename, variables):
    # Read data from ROOT files
    data_sig = ROOT.RDataFrame("Events", signal_filename).AsNumpy()
    data_bkg = ROOT.RDataFrame("Events", background_filename).AsNumpy()

    # Convert inputs to format readable by machine learning tools
    x_sig = np.vstack([data_sig[var] for var in variables]).T
    x_bkg = np.vstack([data_bkg[var] for var in variables]).T
    x = np.vstack([x_sig, x_bkg])
 
    # Create labels
    num_sig = x_sig.shape[0]
    num_bkg = x_bkg.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])

    ### Compute weights balancing both classes
    ##num_all = num_sig + num_bkg
    ##w = np.hstack([np.ones(num_sig) * num_all / num_sig, np.ones(num_bkg) * num_all / num_bkg])

    # Compute weights using existing branch
    w_sig = np.vstack(data_sig["weight"]).T
    w_bkg = np.vstack(data_bkg["weight"]).T 
    w = np.hstack([w_sig, w_bkg]).ravel() 

    return x, y, w

# Set here preferred MVA
useXgboost = False
useTMVABDT = True

hasGPU = ROOT.gSystem.GetFromPipe("root-config --has-tmva-gpu") == "yes"
hasCPU = ROOT.gSystem.GetFromPipe("root-config --has-tmva-cpu") == "yes"

if __name__ == "__main__":

   # NOTE load_data is not working for large sized dataframes due to memory alloc. issues 
   # The workflow is working fine for reduced samples
   if useXgboost:
      # Loop on folds
      for fold in range(0,5):

         print("Training fold ",str(fold))

         # Load data
         x, y, w = load_data(output_folder+"/fold_"+str(fold)+"_signal.root", output_folder+"/fold_"+str(fold)+"_background.root", features)
         #x, y, w = load_data(output_folder+"/fold_"+str(fold)+"_signal_10000evt.root", output_folder+"/fold_"+str(fold)+"_background_10000evt.root", features)

         # Fit xgboost model
         bdt = XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=500)
         bdt.fit(x, y, sample_weight=w)
 
         # Save model in TMVA format
         print("===Training done on ",x.shape[0],"events. Saving model in tmva_fold"+str(fold)+".root")
         ROOT.TMVA.Experimental.SaveXGBoost(bdt, "myBDT", output_folder+"/tmva_xgboost_fold"+str(fold)+".root", num_inputs=x.shape[1])

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
 
      loader = TMVA.DataLoader("dataset")
      input_file_sig = TFile.Open(output_folder+"/fold_0_signal_10M.root")
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
         MaxDepth=8,
         BoostType="RealAdaBoost",
         AdaBoostBeta=0.3,
         UseBaggedBoost=True,
         BaggedSampleFraction=0.1,
         SeparationType="GiniIndex",
         nCuts=-1,
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
