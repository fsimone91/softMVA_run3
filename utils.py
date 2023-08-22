output_folder = "/eos/user/f/fsimone/softMVA_run3"

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
   
   "chi2LocalMomentum",
   "chi2LocalPosition",
   "nStations",
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
                "muonStationsWithValidHits": [0, 10],
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
                "muonStationsWithValidHits": "muonStationsWithValidHits",
                "trkKink":   "log(1+trkKink)",
                "segmentComp": "segmentComp"
               }
