#features
features = [
   "pt",
   "eta",

   "trkLayers",
   "nPixels",
   "trkValidFrac",
   "nValidHits",

   "staNormChi2",
   "trkNormChi2",
   "glbNormChi2",
   "staRelChi2",
   "trkRelChi2",

##globalDeltaEtaPhi missing
##glbKink missing   
   "chi2LocalMomentum",
   "chi2LocalPosition",
   "nStations",
   #"muonStationsWithValidHits",
   "trkKink",
   "segmentComp"
]

spectators = [
   "weight",
   "evt",
   "sim_pdgId",
   "sim_mpdgId",
   "sim_type"
]

#dictionary containing feature names and ranges for binning
feature_dict = {
                "pt": [0, 30],
                "eta": [-2.4, 2.4],

                "trkLayers": [0, 10], #tracker layers
                "nPixels": [0, 10], #number valid pixed hits
                "nValidHits": [0, 35], #number valid hits
                "trkValidFrac": [0,1 ],
      
                "staNormChi2": [0, 20],
                "trkNormChi2": [0, 20],
                "glbNormChi2": [0, 20],
                "staRelChi2": [0, 20],
                "trkRelChi2": [0, 10],
      
                "chi2LocalMomentum": [0, 120],
                "chi2LocalPosition": [0, 10],
                "nStations": [0, 10],
<<<<<<< HEAD
                "muonStationsWithValidHits": [0, 10],
=======
                #"muonStationsWithValidHits": [0, 10],
>>>>>>> softMVA_run3/main
                "trkKink": [0, 35],
                "segmentComp": [0, 1]
               }

#dictionary containing feature names and ranges for training
feature_expression = {
                "pt": "pt",
                "eta": "eta",
                "trkLayers":    "trkLayers", #tracker layers
                "nPixels":      "nPixels", #number valid pixed hits
                "nValidHits":   "nValidHits", #number valid hits
                "trkValidFrac": "trkValidFrac",
      
                "staNormChi2": "staNormChi2>50?50:staNormChi2", 
                "trkNormChi2": "trkNormChi2>50?50:trkNormChi2", 
                "glbNormChi2": "glbNormChi2>50?50:glbNormChi2", 
                "staRelChi2":  "staRelChi2>50?50:staRelChi2", 
                "trkRelChi2":  "trkRelChi2>50?50:trkRelChi2", 
      
                "chi2LocalMomentum": "chi2LocalMomentum>150?150:chi2LocalMomentum",
                "chi2LocalPosition": "chi2LocalPosition>50?50:chi2LocalPosition",
                "nStations": "nStations",

                #"muonStationsWithValidHits": "muonStationsWithValidHits",
                "trkKink":   "log(1+trkKink)<3?3:log(1+trkKink)",
                "segmentComp": "segmentComp"
               }

#dictionary containing feature names in tau3mu ntuples
feature_dict_tau3mu = {

          "pt":               "mu_pt",            
          "eta":              "mu_eta",
                              
          "trkLayers":        "mu_trackerLayersWithMeasurement",
          "nPixels":          "mu_Numberofvalidpixelhits",
          "trkValidFrac":     "mu_innerTrack_validFraction",
          "nValidHits":       "mu_GLhitPattern_numberOfValidMuonHits",
                                                                 
          "staNormChi2":      "mu_outerTrack_normalizedChi2",
          "trkNormChi2":      "mu_innerTrack_normalizedChi2",
          "glbNormChi2":      "mu_GLnormChi2",
          "staRelChi2":       "mu_combinedQuality_trkRelChi2",
          "trkRelChi2":       "mu_combinedQuality_staRelChi2",
                              
          "chi2LocalMomentum":"mu_combinedQuality_chi2LocalMomentum",
          "chi2LocalPosition":"mu_combinedQuality_chi2LocalPosition",
          "nStations":        "mu_numberOfMatchedStations",
          "trkKink":          "mu_combinedQuality_trkKink",
          "segmentComp":      "mu_segmentCompatibility"
}
