import pandas as pd
import numpy as np
import glob
from scipy.stats import skew, kurtosis

# Path to your folder
path = r"C:\Users\user\Desktop\Adjusted_data"   # <-- change to yours
files = glob.glob(path + "/*.xlsx")

sheets = ["BL", "AML", "APL1", "APL2", "meal", "evening", "3AM"]

print("Total volunteers found:", len(files))





def extract_features(df):
    if "GSR" not in df.columns:
        return None

    gsr = df["GSR"].dropna().values

    if len(gsr) == 0:
        return None

    features = [
        np.mean(gsr),
        np.median(gsr),
        np.std(gsr),
        np.min(gsr),
        np.max(gsr),
        skew(gsr),
        kurtosis(gsr),
        np.percentile(gsr, 75) - np.percentile(gsr, 25)  # IQR
    ]
    return features





all_features = {sheet: [] for sheet in sheets}

for f in files:
    print("\n📄 Processing:", f)

    for sheet in sheets:
        try:
            df = pd.read_excel(f, sheet_name=sheet)
        except:
            print(f"❌ Missing sheet: {sheet} in {f}")
            continue

        feats = extract_features(df)
        
        if feats is not None:
            all_features[sheet].append(feats)
        else:
            print(f"⚠️ No valid GSR data in {sheet} for {f}")





from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_pca(group1, group2, name1="A", name2="B"):
    G1 = np.array(all_features[group1])
    G2 = np.array(all_features[group2])

    # Labels for PCA plot
    labels = np.array([name1]*len(G1) + [name2]*len(G2))

    X = np.vstack([G1, G2])

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(X_scaled)

    print(f"\nPCA: {group1} vs {group2}")
    print("Explained variance:", pca.explained_variance_ratio_)

    return pc, labels



pc, labels = run_pca("BL", "AML", "BL", "AML")
