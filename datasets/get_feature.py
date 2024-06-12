def extract_features(file_path):
    import opensmile
    import numpy as np
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = np.array(smile.process_file(file_path)).reshape(-1)
    
    return features

if __name__ == "__main__":
    file_path = "data/emodb/03a01Nc.wav"
    print(extract_features(file_path).shape)