#type: ignore

import pandas as pd
import numpy as np
import sklearn
from scipy.stats import boxcox, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.decomposition import PCA

df1 = pd.read_excel(r"D:\Final_master_sheet.xlsx", sheet_name='APL2', header=0)
df1["label"] = 1
df1.iloc[0:1]
df1.columns
df1.drop(['Unnamed: 26'], axis=1, inplace=True)
df1.rename(columns={'Overall_Average':'avg_blink_dur'}, inplace=True)
print(df1)
df1


df2 = pd.read_excel(r"D:\Final_master_sheet.xlsx", sheet_name='BL', header=0)
df2.rename(columns={'Overall_Average':'avg_blink_dur'}, inplace=True)
df2["label"] = 0
print(df2)
volunteers_to_remove = ['V51', 'V54', 'V55', 'V57', 'V58']
volunteer_col = df2.columns[0]
df2 = df2[~df2[volunteer_col].isin(volunteers_to_remove)]
df2.columns
df2.isna().sum()
numeric_cols = df2.select_dtypes(include=[np.number]).columns
df_no_nan = df2[numeric_cols].fillna(df2[numeric_cols].median())

df_no_nan['Volunteers'] = df2['Volunteers']
df_no_nan.columns
volunteers_to_remove = ['V51', 'V54', 'V55', 'V57', 'V58']
volunteer_col = df1.columns[0]
df_1 = df1[~df1[volunteer_col].isin(volunteers_to_remove)]
df_1.isna().sum()
df_1.columns
df_1.isna().sum()
df_no_nan1 = df_1.dropna()
numeric_c1 = df_no_nan1.select_dtypes(include=[np.number]).columns
df_no_nan1[numeric_c1].skew()
df_1.shape

combined_df  = pd.concat([df_no_nan, df_1], ignore_index=False)
combined_df['avg_blink_dur'].skew()
combined_df.shape
combined_df.isna().sum()
combined_df.drop(['AUDIO', 'MEANING', 'COLOR'], axis=1, inplace=True)

shuffled_data = combined_df.sample(frac=1, random_state=42).reset_index(drop=False)
shuffled_data.shape
print(shuffled_data['label'].value_counts())
print(shuffled_data.head())
shuffled_data.describe()

from sdv.metadata import SingleTableMetadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=shuffled_data)  
#metadata.set_primary_key("Volunteers")
metadata.update_column('Volunteers', sdtype='categorical') 
print(metadata.to_dict())
report = metadata.validate()
metadata.validate_data(shuffled_data)  # Validates against shuffled_data
print(report)

print(shuffled_data['label'].value_counts()) 
print(shuffled_data['label'].nunique()) 
#
df_no_nan[numeric_c1].skew()
df_no_nan[numeric_c1].kurtosis()

numeric_c1 = df_no_nan.select_dtypes(include=[np.number]).columns
df_no_nan[numeric_c1].skew()
df_no_nan[numeric_c1].kurtosis()
#

from sdv.single_table import CTGANSynthesizer
synthesizer = CTGANSynthesizer(
    metadata,
    enforce_rounding=True,
    epochs=10000,
    enforce_min_max_values=True, 
    embedding_dim=64,           
    generator_dim=(256, 256),  
    discriminator_dim=(256, 128),                              
    discriminator_steps=5,     
    generator_lr=1e-5,          
    pac=25,
    enable_gpu=False,
    batch_size=200,
    log_frequency=True,
    verbose=True,
)
synthesizer.get_parameters()
print(synthesizer.get_parameters())
synthesizer = CTGANSynthesizer(metadata)

synthesizer.fit(shuffled_data)

synthetic_data = synthesizer.sample(num_rows=8000)
synthetic_data.shape
synthetic_data['label'].value_counts()
synthetic_data['Volunteers'].value_counts()

synthesizer.get_loss_values()
losses_df = synthesizer.get_loss_values()
print(losses_df.tail(100))
losses_df.plot(x='Epoch', y=['Generator Loss', 'Discriminator Loss'])
plt.show()  # Matplotlib 

real_vols = [f'V{i}' for i in range(1, 62)]
real_vol_data = shuffled_data[shuffled_data['Volunteers'].isin(real_vols)].copy()
print("Real V1-V61 rows:", real_vol_data.shape[0])

print("Synthetic base shape:", synthetic_data.shape)

n_synth_vols = 1939
synth_vols = [f'V{i}' for i in range(62, 62 + n_synth_vols)]
synth_vol_data = []

np.random.seed(42)  
for vol in synth_vols:
    vol_rows = synthetic_data.sample(n=2, replace=False, random_state=np.random.randint(4000)).copy()
    vol_rows['Volunteers'] = vol
    synth_vol_data.append(vol_rows)

synth_vol_df = pd.concat(synth_vol_data, ignore_index=True)
print("Synthetic V62-V1000 shape:", synth_vol_df.shape)
synth_vol_df['Volunteers'].value_counts()

from sdv.evaluation.single_table import evaluate_quality
quality_report = evaluate_quality(shuffled_data, synthetic_data, metadata)
print(quality_report)
print(synthetic_data.min())
print(synthetic_data['Blinks count'].max())
print(synthetic_data['Volunteers'].value_counts())
print(synthetic_data['label'].value_counts())
synthetic_data.info()
print(synthetic_data.tail(10))
synth_vol_df.shape
synth_vol_df['label'].value_counts()
shuffled_data['label'].value_counts()
from sdv.evaluation.single_table import get_column_plot

ql_df.plot = get_column_plot(
    real_data=shuffled_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='label'
)
    
plt.show()



final_data = pd.concat([shuffled_data, synth_vol_df], ignore_index=True)
final_data['label'].value_counts()
final_data[final_data['Volunteers'] == 'V100']

final_data.head()
print("\n=== FINAL RESULTS ===")
print("Total rows:", final_data.shape[0])
print("Total unique volunteers:", final_data['Volunteers'].value_counts())
print("Labels:", final_data['label'].value_counts().to_dict())
final_data.to_excel(r"F:\final1synthetic_drowsiness_data.xlsx", index=True)


###########with unique values of volunteers#######

df3 = pd.read_excel(r"F:\final1synthetic_drowsiness_data.xlsx", header=0)
df3.head()
df3.drop(['Unnamed: 0'], axis=1, inplace=True)
df3.shape
df3.isnull().sum()
df3['label'].nunique()
df3['Volunteers'].nunique()
df3['label'].value_counts()
df3.drop(['index'], axis=1, inplace=True)
df3.tail()
df3.describe()
df3

numeric_c3 = df3.select_dtypes(include=[np.number]).columns
numeric_c3.isnull().sum()
df3[numeric_c3].skew()
df3[numeric_c3].kurtosis()
from sklearn.preprocessing import StandardScaler

exclude_cols = ['Volunteers', 'label'] 
standard_features = [f for f in numeric_c3 if f not in exclude_cols]
df_transformed = df3[standard_features].copy()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_transformed[standard_features])
df_scaled = pd.DataFrame(df_scaled, columns=df_transformed.columns)
df_scaled = pd.concat(
    [df_scaled, df3[['Volunteers', 'label']]],
    axis=1
)
df_scaled


from scipy.stats import zscore
import pandas as pd
import numpy as np
features = ['Blinks count', 'avg_blink_dur']
df_clean = df3.copy()
df_clean[features] = df_clean[features].astype(float)
df_clean[df_clean['label'] == 0][features].dropna() 
z_scores = zscore(df_clean[features], nan_policy='omit')
print(z_scores)

z_df = pd.DataFrame(
    z_scores,
    columns=features,
    index=df_clean.index
)
zthresh = 1.64
mask = (z_df >= 1.64) | (z_df <= -1.64)

df_clean[features] = df_clean[features].mask(mask, np.nan)
print(mask.dtypes)
z_outliers = z_df[mask]
print("Outliers detected:\n", z_outliers)

df_clean['Blinks count'] = (
    df_clean['Blinks count']
    .fillna(df_clean['Blinks count'].median())
    .round()
    .astype(int)
)

df_clean['avg_blink_dur'] = (
    df_clean['avg_blink_dur']
    .fillna(df_clean['avg_blink_dur'].median())
)
df_clean[features].shape
df_clean['Blinks count'].skew()
df_clean['avg_blink_dur'].skew()
feat = df_clean['avg_blink_dur']


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True)

skew_cols = ['std_phasic', 'mean_phasic', 'LF', 'HF', 'avg_blink_dur', 'Blinks count']
df_clean[skew_cols] = pt.fit_transform(df_clean[skew_cols])

print(df_clean[skew_cols].skew().sort_values(key=abs))  # |skew| <0.5




import matplotlib.pyplot as plt
import seaborn as sns

df_corri = df_scaled.drop(['Volunteers', 'label'], axis=1)
corr_matrix = df_corri.corr(method='spearman')
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            cbar_kws={'shrink': 0.8},
            xticklabels=True,
            yticklabels=True)
plt.title('Correlation Heatmap (28 Features)', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

corr_matrix = df_corri.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            cbar_kws={'shrink': 0.8},
            xticklabels=True,
            yticklabels=True)
plt.title('Correlation Heatmap (28 Features)', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()
high_corr = corr_matrix[(np.abs(corr_matrix) > 0.8) & (corr_matrix != 1.0)].stack().reset_index()
high_corr.columns = ['Feature1', 'Feature2', 'Correlation']
high_corr = high_corr.sort_values('Correlation', key=abs, ascending=False)
print("High correlations (|r| > 0.8):")
print(high_corr.head(10))
top_pairs = high_corr.head(5)[['Feature1', 'Feature2']].values

from sklearn.decomposition import PCA
#df_scaled2.columns
numeric_cols = df_scaled.select_dtypes(include=['number']).columns.drop('label')
X1 = df_scaled[numeric_c3].values
y1 = df_scaled['label'].values
df_scaled.shape
df_scaled.isnull().sum()
colors = ['blue', 'orange']
label_names = ['0', '1']
pca = PCA()
X1_pca = pca.fit_transform(X1)
plt.figure(figsize=(8,6))
for label in [0, 1]:
    plt.scatter(X1_pca[y1 == label, 1], X1_pca[y1 == label, 2],
                c=colors[label], label=label_names[label], alpha=0.7)
plt.title('PCA 2D projection')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True)
plt.show()

df_scaled1.shape
df_scaled1.isnull().sum()
from sklearn.decomposition import PCA

X1 = df_scaled1[numeric_c3].values
y1 = df_scaled1['label'].values

colors = ['blue', 'orange']
label_names = ['0', '1']
pca = PCA()
X1_pca = pca.fit_transform(X1)
plt.figure(figsize=(8,6))
for label in [0, 1]:
    plt.scatter(X1_pca[y1 == label, 1], X1_pca[y1 == label, 2],
                c=colors[label], label=label_names[label], alpha=0.7)
plt.title('PCA 2D projection')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True)
plt.show()