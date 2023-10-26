import ROOT
from ROOT import RDataFrame

import numpy as np
import pickle

<<<<<<< HEAD
from utils import *
 
# Enable multi-threading
#ROOT.ROOT.EnableImplicitMT()
=======
import psutil
import time

from utils import *
 
# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()
>>>>>>> softMVA_run3/main

# Batch mode
ROOT.gROOT.SetBatch(ROOT.kTRUE)

# JIT a C++ function from Python
ROOT.gInterpreter.Declare("""
double get_weight(const TH2D* _h, const double pt, const double eta ){
    double w = 1.0;
    if( pt < _h->GetXaxis()->GetXmax() && pt > _h->GetXaxis()->GetXmin() &&
        std::abs(eta) < _h->GetYaxis()->GetXmax() && std::abs(eta) > _h->GetYaxis()->GetXmin()){
        int ipt = _h->GetXaxis()->FindBin(pt);
        int ieta = _h->GetYaxis()->FindBin(std::abs(eta));
        w = _h->GetBinContent(ipt,ieta);
        //err = _h->GetBinError(ipt,ieta);
    }
    return w;
}
double get_weighterr(const TH2D* _h, const double pt, const double eta ){
    double we = 0.0;
    if( pt < _h->GetXaxis()->GetXmax() && pt > _h->GetXaxis()->GetXmin() &&
        std::abs(eta) < _h->GetYaxis()->GetXmax() && std::abs(eta) > _h->GetYaxis()->GetXmin()){
        int ipt = _h->GetXaxis()->FindBin(pt);
        int ieta = _h->GetYaxis()->FindBin(std::abs(eta));
        we = _h->GetBinError(ipt,ieta);
    }
    return we;
}

// snippet of code to compute weights if the pt and eta columns contain vectors
//ROOT::RVecF compute_weights(const ROOT::RVecF &pt, const ROOT::RVecF &eta, TH2D &h) {
//   auto mySize = pt.size();
//   ROOT::VecOps::RVec<float> outputArray(mySize);
//   for (size_t iii = 0; iii < mySize; iii++){
//      outputArray.at(iii) = get_weight(&h, pt[iii], eta[iii]);
//   }
//   return outputArray;
//}
//
//struct WeightsComputer {
//   TH2D *fHist2D;
//   WeightsComputer(TH2D *h) : fHist2D(h) {}
//
//   ROOT::RVecF operator()(const ROOT::RVecF &pt, const ROOT::RVecF &eta) {
//      return compute_weights(pt, eta, *fHist2D);
//  }
//};

struct WeightsComputer {
   TH2D *fHist2D;
   WeightsComputer(TH2D *h) : fHist2D(h) {}

   float operator()(const float pt, const float eta) {
      return get_weight(fHist2D, pt, eta);
  }
};
""")

def list_rootfiles(path):
   #generating the list of all .root files in given directory and subdirectories
   file_list = []
   for r, d, f in os.walk(path): # r=root, d=directories, f = files
       for file in f:
           if '.root' in file:
               file_list.append(os.path.join(r, file))

   return file_list

def call_th1(df1, df2, f, w):
       print("plotting feature ",f)
       #binning from dictionary
       x1 = feature_dict[f][0]
       x2 = feature_dict[f][1]
       h1 = df1.Histo1D((f, f, 50, x1, x2), f, w)
       h2 = df2.Histo1D((f, f, 50, x1, x2), f, w)
       return h1, h2

<<<<<<< HEAD
def draw_th1(h1, h2, f, w, label1, label2, c):
=======
def draw_th1(h1, h2, f, label1, label2, c, name):
>>>>>>> softMVA_run3/main
       h1.Scale(1.0/h1.Integral())
       h2.Scale(1.0/h2.Integral())

       h1.SetLineColor(2)
       h1.Draw("hist")
       h2.Draw("hist same")

       leg = ROOT.TLegend(.73,.32,.97,.53)
       leg.SetBorderSize(0)
       leg.SetFillColor(0)
       leg.SetFillStyle(0)
       leg.SetTextFont(42)
       leg.SetTextSize(0.035)
       leg.AddEntry(h1.GetPtr(),label1,"L")
       leg.AddEntry(h2.GetPtr(),label2,"L")
       leg.Draw()

<<<<<<< HEAD
       c.SaveAs("plots/"+f+"_weighted.png")
=======
       c.SaveAs(output_folder+"/plots/"+f+"_"+name+".png")
>>>>>>> softMVA_run3/main

def nano_to_DF(path,treename):
   frame = RDataFrame(treename, path+"/*.root")
   return frame

def fold_df(df, label, nfolds, columns):
   #columns = ROOT.std.vector["string"](list(feature_dict.keys()))

   # Split dataset by event number for training and testing
   for f in range(0,nfolds):
      fold = str(f)
      df.Filter("evt % "+str(nfolds)+" == "+fold, "Select events for fold "+fold)\
<<<<<<< HEAD
        .Snapshot("Events", "fold_" + fold + "_"+label+".root", columns)
=======
        .Snapshot("Events", output_folder+"/fold_" + fold + "_"+label+".root", columns)
>>>>>>> softMVA_run3/main

def prepare_data_numpy(df_sig, df_bkg, variables):

    # Convert inputs to format readable by machine learning tools
    x_sig = np.vstack([df_sig[var] for var in variables]).T
    x_bkg = np.vstack([df_bkg[var] for var in variables]).T
    x = np.vstack([x_sig, x_bkg])
 
    # Create labels
    num_sig = x_sig.shape[0]
    num_bkg = x_bkg.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
 
    # Compute weights balancing both classes
    #num_all = num_sig + num_bkg
    #w = np.hstack([np.ones(num_sig) * num_all / num_sig, np.ones(num_bkg) * num_all / num_bkg])

    # Compute weights using existing branch
    w_sig = np.vstack(df_sig["weight"]).T 
    w_bkg = np.vstack(df_bkg["weight"]).T 
    w = np.vstack([w_sig, w_bkg])
    return x, y, w

if __name__ == "__main__":
<<<<<<< HEAD
   #path = "/eos/cms/store/group/phys_bphys/bmm/bmm6/PostProcessing/FlatNtuples/523/muon_mva/InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/"
   path = "/eos/cms/store/group/phys_bphys/bmm/bmm6/PostProcessing/FlatNtuples/524/muon_mva/InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/"
   #df = nano_to_DF(path,"muons")
   df = nano_to_DF(path,"muons").Range(10000000) #Note: range is not compatible with MT
=======

   start = time.time()

   #path = "/eos/cms/store/group/phys_bphys/bmm/bmm6/PostProcessing/FlatNtuples/523/muon_mva/InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/"
   path = "/eos/cms/store/group/phys_bphys/bmm/bmm6/PostProcessing/FlatNtuples/524/muon_mva/InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/"
   df = nano_to_DF(path,"muons")
   #df = nano_to_DF(path,"muons").Range(100000000) #Note: range is not compatible with MT
>>>>>>> softMVA_run3/main
   
   #selections
   acceptance = "(pt>3.5 && abs(eta)<1.2) || (pt>2.0 && abs(eta)>1.2 && abs(eta)<2.4)"
   basic_quality_cuts = "(highPurity && isGlobal && glbTrackProbability>0 && chargeProduct>0)" 
   pion = "(abs(sim_pdgId) == 211)"
   kaon = "(abs(sim_pdgId) == 321)"
   pi_mu_decay = "(abs(sim_pdgId) == 13 && abs(sim_mpdgId) == 211 && sim_type == 1)"
   k_mu_decay  = "(abs(sim_pdgId) == 13 && abs(sim_mpdgId) == 321 && sim_type == 1)"
<<<<<<< HEAD
   muon = "abs(sim_pdgId) == 13 && sim_type>1"
=======
   muon = "abs(sim_pdgId) == 13 && sim_type==3"
>>>>>>> softMVA_run3/main
   
   df = df.Filter(acceptance)
   df = df.Filter(basic_quality_cuts)
   df = df.Define("abs_eta", "abs(eta)")
   
   entries = df.Count()
   print("total ",entries.GetValue())
   
   # Signal selections
   df_sig = df.Filter(muon, "signal")
   
   # Background selections
   df_bkg = df.Filter(pion +"||"+ kaon +"||"+ pi_mu_decay +"||"+ k_mu_decay, "background")
   
   #entries_s = df_sig.Count()
   #entries_b = df_bkg.Count()
   #print("signal ",entries_s.GetValue())
   #print("bkg    ",entries_b.GetValue())
   #report = df.Report()
   #report.Print()

   #in this section, we plot pt and eta and draw the ratio plot for reweighting 
   signal_pt = df_sig.Histo1D(("pt", "signal pt", 60, 0, 30), "pt");
   bkg_pt    = df_bkg.Histo1D(("pt", "bkg pt", 60, 0, 30), "pt");
   signal_eta = df_sig.Histo1D(("abs(eta)", "signal eta", 48, 0, 2.4), "abs_eta");
   bkg_eta    = df_bkg.Histo1D(("abs(eta)", "bkg eta", 48, 0, 2.4), "abs_eta");
   
   signal_pt_eta = df_sig.Histo2D(("s_pt_eta", "signal pt-eta", 30, 0, 30.0, 24, 0, 2.4), "pt", "abs_eta");
   bkg_pt_eta    = df_bkg.Histo2D(("b_pt_eta", "bkg pt-eta",    30, 0, 30.0, 24, 0, 2.4), "pt", "abs_eta");

   canvas = ROOT.TCanvas("c", "c", 800, 800)
   canvas.Divide(2, 2)
   upleft_pad = canvas.cd(1)
   signal_pt.Draw("Hist")
   upright_pad = canvas.cd(2)
   bkg_pt.Draw("Hist")
   upleft_pad = canvas.cd(3)
   signal_eta.Draw("Hist")
   upright_pad = canvas.cd(4)
   bkg_eta.Draw("Hist")
<<<<<<< HEAD
   canvas.SaveAs("plots/muon_eta_pt.png")
=======
   canvas.SaveAs(output_folder+"/plots/muon_eta_pt.png")
>>>>>>> softMVA_run3/main
   
   #plot
   rcanvas = ROOT.TCanvas("c", "c", 800, 800)
   signal_pt_eta.Draw("colz")
<<<<<<< HEAD
   rcanvas.SaveAs("plots/signal_eta_pt.png")
   bkg_pt_eta.Draw("colz")
   rcanvas.SaveAs("plots/bkg_eta_pt.png")
=======
   rcanvas.SaveAs(output_folder+"/plots/signal_eta_pt.png")
   bkg_pt_eta.Draw("colz")
   rcanvas.SaveAs(output_folder+"/plots/bkg_eta_pt.png")
>>>>>>> softMVA_run3/main
   
   #normalise to same entries
   signal_pt_eta.Scale(1.0/signal_pt_eta.Integral())
   bkg_pt_eta.Scale(1.0/bkg_pt_eta.Integral())
   
   #ratio plot for reweighting
   r_pt_eta = signal_pt_eta.Clone()
   r_pt_eta.Divide(bkg_pt_eta.GetPtr())
   
   r_pt_eta.Draw("colz")
<<<<<<< HEAD
   rcanvas.SaveAs("plots/ratio_eta_pt.png")
=======
   rcanvas.SaveAs(output_folder+"/plots/ratio_eta_pt.png")
>>>>>>> softMVA_run3/main

   #apply reweighting
   df_bkg = df_bkg.Define("weight", ROOT.WeightsComputer(r_pt_eta), ["pt", "eta"])
   df_sig = df_sig.Define("weight", "1.0")

   #print(df_bkg.GetColumnNames())
   #print(df_sig.GetColumnNames())

   #plot interesting features
   c2 = ROOT.TCanvas("c2", "c2", 600, 800)
   histo_pairs = []
   for f in features:
       histo_pairs.append(call_th1(df_sig, df_bkg, f, "weight"))
   for index, histo_pair in enumerate(histo_pairs):
<<<<<<< HEAD
       draw_th1(histo_pair[0], histo_pair[1], features[index], "weight", "signal", "bkg", c2)
       
   print("performed ",df_sig.GetNRuns()," loops")

   #training

   # from RDataFrame to numpy arrays for training
   #x, y, w = prepare_data_numpy(df_sig, df_bkg, features+spectators)
  
   # from big RDataFrame to folds containing only needed features 
   fold_df(df_sig, "signal_10M", 5, features+spectators)
   fold_df(df_bkg, "background_10M", 5, features+spectators)
=======
       draw_th1(histo_pair[0], histo_pair[1], features[index], "signal", "bkg", c2, "weighted")
       
   print("performed ",df_sig.GetNRuns()," loops")

   # from big RDataFrame to folds containing only needed features 
   fold_df(df_sig, "signal_jpsi_10M", 5, features+spectators)
   fold_df(df_bkg, "background_10M", 5, features+spectators)

   #monitor execution time
   end = time.time()
   print('execution time ', end-start)
   #monitor CPU usage
   print('CPU usage ', psutil.cpu_percent())
   #monitoring memory usage
   print('memory usage ', psutil.virtual_memory())
>>>>>>> softMVA_run3/main
