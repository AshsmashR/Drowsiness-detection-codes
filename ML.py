corr_matrix = df_transformed.corr() ###used your scaled df here not exactly this
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

# 3) Plot each pair colored by label
for f1, f2 in top_pairs:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=combined_df,
        x=f1,
        y=f2,
        hue='label',           # color by class label
        palette='coolwarm',
        alpha=0.7
    )
    plt.title(f'{f1} vs {f2} colored by label')
    plt.tight_layout()
    plt.show()
#high coorrelations exists between mean_EDA and mean_tonic, std_EDA and std_tonic, RMSSD and SD1, SDRR and SD2   
   
#step2 
##now will see mutual information scores
X = df_transformed.drop('label', axis=1).fillna(0)  # Fill NaNs for MI
y = df_transformed['label']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)
print("Top 10 features by Mutual Information:")
print(mi_scores_df.head(10))

# Select top K features by MI
selector = SelectKBest(mutual_info_classif, k=8)
X_mi_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"\nTop 8 MI features: {selected_features}")
plt.figure(figsize=(12, 8))
top_mi = mi_scores_df.head(8)
sns.barplot(data=top_mi, x='MI_Score', y='Feature', palette='viridis')
plt.title('Top 8 Features by Mutual Information Score')
plt.xlabel('MI Score')
plt.tight_layout()
plt.show()


####PCA analysis#####
from sklearn.decomposition import PCA

numeric_cols = df_transformed.select_dtypes(include=['number']).columns.drop('label')
X = df_transformed[numeric_cols].values
y = df_transformed['label'].values

colors = ['blue', 'orange']
label_names = ['0', '1']
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8,6))
for label in [0, 1]:
    plt.scatter(X_pca[y == label, 1], X_pca[y == label, 2],
                c=colors[label], label=label_names[label], alpha=0.7)
plt.title('PCA 2D projection')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True)
plt.show()
##
pca2 = PCA(n_components=X.shape[1])
pca2.fit(X)
explained_variance = pca2.explained_variance_ratio_
print('Explained variance ratio per PC:', explained_variance)
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.grid(True)
plt.show()
##
loadings = pd.DataFrame(pca2.components_.T, columns=[f'PC{i}' for i in range(1, X.shape[1]+1)],
                        index=numeric_cols)
print('Feature loadings on first few PCs:')
print(loadings.iloc[:, :5])  
##
X_pca3 = PCA(n_components=6)
X_pca3 = X_pca3.fit_transform(X)
colors = ['blue', 'orange']
label_names = ['0', '1']
color_vals = [colors[label] for label in y]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=color_vals, alpha=0.7)
ax.set_title('3D PCA Visualization')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.show()

##PER PC COMPONENTS 
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(6)], index=X.columns)
print("Top 5 features per PC component:")
for i in range(6):
    top_features = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5)
    print(f"\nPC{i+1} top features:\n", top_features)
print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance: {pca.explained_variance_ratio_.cumsum()}")

plt.figure(figsize=(12, 10))
sns.heatmap(loadings, annot=False, cmap='RdBu_r', center=0, 
            xticklabels=[f'PC{i+1}' for i in range(6)])
plt.title('Feature Loadings in 6 PC Components')
plt.tight_layout()
plt.show()
##
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.ravel()
for i in range(6):
    top_pc = loadings[f'PC{i+1}'].abs().sort_values(ascending=True).tail(8)
    axes[i].barh(range(len(top_pc)), top_pc.values, color='skyblue')
    axes[i].set_yticks(range(len(top_pc)))
    axes[i].set_yticklabels(top_pc.index)
    axes[i].set_title(f'PC{i+1} (Var: {pca.explained_variance_ratio_[i]:.2%})')
    axes[i].set_xlabel('Loading')
plt.tight_layout()
plt.show()

#LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)
print(f"LDA explained variance ratio: {lda.explained_variance_ratio_}")
y_pred_lda = lda.predict(X)
accuracy = accuracy_score(y, y_pred_lda)
print(f"LDA Classification Accuracy: {accuracy:.4f}")
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_lda, y, c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel('LD1 (100% Class Variance)')
plt.title(f'LDA: Separation (Accuracy: {accuracy:.1%})')
plt.yticks([0, 1], ['BL (0)', 'APL2 (1)'])
plt.show()

features = numeric_df.drop('label', axis=1).columns
lda_loadings = pd.DataFrame({
    'Feature': features,
    'LDA_Loading': np.abs(lda.coef_[0])
}).sort_values('LDA_Loading', ascending=False)
print("\nTop 15 LDA features:")
print(lda_loadings.head(10))

########need  to see pca coorelations with lda components#######
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(6)])
X_lda_df = pd.DataFrame(X_lda, columns=['LD1'])
pca_lda_combined = pd.concat([X_pca_df, X_lda_df], axis=1)
pca_lda_corr = pca_lda_combined.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(pca_lda_corr, 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            fmt='.3f',
            square=True)
plt.title('PCA Components vs LDA Component Correlation')
plt.tight_layout()
plt.show()
print("PCA vs LDA correlations:")
print(pca_lda_corr['LD1'].sort_values(key=abs, ascending=False))
pcs_to_use = ['PC1', 'PC2', 'PC6']  # PCs correlated with LD1
top_features_per_pc = {}
for pc in pcs_to_use:
    # absolute loading strength
    top_features_per_pc[pc] = (
        loadings[pc]
        .abs()
        .sort_values(ascending=False)
        .head(8)              # top 8 per PC (tune as you like)
        .index
        .tolist()
    )

top_features_per_pc
candidate_features = sorted(set(
    f for feats in top_features_per_pc.values() for f in feats
))
feature_scores = (
    loadings.loc[candidate_features, pcs_to_use]
    .abs()
    .sum(axis=1) 
    .sort_values(ascending=False)
)
selected_features = feature_scores.head(15).index.tolist()   # pick top 10–20
print("Final selected features:\n", selected_features)
##VIF 

X_vif = df_transformed[numeric_cols].values
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_cols
vif_data["VIF"] = [variance_inflation_factor(X_vif, i) for i in range(len(numeric_cols))]
print("VIF before transformation:\n", vif_data)
plt.figure(figsize=(10, 6))
sns.barplot(x='VIF', y='Feature', data=vif_data.sort_values('VIF', ascending=False), palette='viridis')
plt.title('Variance Inflation Factor (VIF) Before Transformation')  
plt.xlabel('VIF')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

feat_names = sorted(set(mi_scores_df['Feature'])
                    | set(loadings.index)
                    | set(lda_loadings['Feature']))

score_df = pd.DataFrame(index=feat_names)
score_df['MI'] = mi_scores_df.set_index('Feature')['MI_Score']
score_df['PCA_score'] = loadings.loc[feat_names, ['PC1','PC2','PC3','PC4','PC5','PC6']].abs().sum(axis=1)
score_df['LDA'] = lda_loadings.set_index('Feature')['LDA_Loading'].abs()
score_df = (score_df - score_df.min()) / (score_df.max() - score_df.min())

score_df['TOTAL'] = 0.4*score_df['MI'] + 0.3*score_df['PCA_score'] + 0.3*score_df['LDA']
to_drop = ['mean_EDA','mean_tonic','RMSSD','SD1','SDRR','SD2'] 
score_df = score_df.drop(index=to_drop, errors='ignore')
best_feats = score_df.sort_values('TOTAL', ascending=False).head(12)
print(best_feats)



##SELECTION AFTER THAT  ###############3select it accordingly 
selected_features = ["
#"AUDIO",
#"MEANING" ##with respect of PCA LDA analysis and Mutual information scores AND VIF.
#REMOVED AUDIO & MEANING 

new_df = df_transformed[selected_features]
print(new_df.head())
new_df.columns

X = new_df
y = df_transformed['label']
X = new_df[selected_features].copy()
X = X.fillna(X.median())
X.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train.shape, X_test.shape
cv = StratifiedKFold(
    n_splits=10,        # change to 10 if you want 10-fold CV
    shuffle=True,
    random_state=42
)
loo = LeaveOneOut()
knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
    
])
knn_param_grid = {
    
    'clf__n_neighbors': [3, 5, 7, 9, 11],
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan', 'minkowski']
}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid, cv=loo, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
print("KNN best params:", knn_grid.best_params_)
print("KNN best CV acc:", knn_grid.best_score_)
y_pred_knn = knn_grid.predict(X_test)
print("KNN test acc:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

############RF############
rf = RandomForestClassifier(random_state=42,class_weight='balanced')
rf_param_grid = {
    'criterion': ['entropy', 'gini'],        # tune/use entropy
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    
}

rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv,  scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)


print("RF best params:", rf_grid.best_params_)
print("RF best CV acc:", rf_grid.best_score_)
y_pred_rf = rf_grid.predict(X_test)
print("RF test acc:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


#tree based feature importance
best_rf = rf_grid.best_estimator_

rf_importances = best_rf.feature_importances_
rf_imp_df = pd.DataFrame({
    'Feature': selected_features,         
    'Importance': rf_importances
}).sort_values('Importance', ascending=False)

print(rf_imp_df)

plt.figure(figsize=(8, 6))
plt.barh(rf_imp_df['Feature'], rf_imp_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

############
rf2 = RandomForestClassifier(criterion='entropy', max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=10, n_estimators=500, random_state=42, class_weight= None)
rf2.fit(X_train, y_train)
y_pred_rf2 = rf2.predict(X_test)
print("RF2 test acc:", accuracy_score(y_test, y_pred_rf2))
print(classification_report(y_test, y_pred_rf2))

#######################3
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.5, 1]
}

xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

print("XGB best params:", xgb_grid.best_params_)
print("XGB best CV acc:", xgb_grid.best_score_)
y_pred_xgb = xgb_grid.predict(X_test)
print("XGB test acc:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))



best_xgb = xgb_grid.best_estimator_
xgb_importances = best_xgb.feature_importances_
xgb_imp_df = pd.DataFrame({
    'Feature': selected_features,          # your selected feature list
    'Importance': xgb_importances
}).sort_values('Importance', ascending=False)

print(xgb_imp_df)

plt.figure(figsize=(8, 6))
plt.barh(xgb_imp_df['Feature'], xgb_imp_df['Importance'])
plt.gca().invert_yaxis()
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
 

##SVM

svc = SVC(probability=True, random_state=42)


# Random search space around your params
param_distributions = {
    'C': loguniform(1e-3, 1e3),
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5],          # used only if kernel='poly'
    'coef0': uniform(0, 1),         # for poly/sigmoid
    'class_weight': [None, 'balanced']
}
random_search_svc = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_distributions,
    n_iter=50,                # increase for more thorough search
    scoring='accuracy',
    cv=cv,                     # or your StratifiedKFold/LOO
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_svc.fit(X_train, y_train)
y_pred_svc = random_search_svc.predict(X_test)
print("SVC test acc:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
print("Best params:", random_search_svc.best_params_)
print("Best CV score:", random_search_svc.best_score_)


#############

from sklearn.ensemble import VotingClassifier
best_knn = knn_grid.best_estimator_
best_rf  = rf_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_
best_svc = random_search_svc.best_estimator_

soft_voting_clf = VotingClassifier(
    estimators=[
        ('knn', best_knn),
        ('rf',  best_rf),
        ('xgb', best_xgb),
        ('svc', best_svc)
    ],
    voting='soft', 
    weights=[2,2,1,1],
    n_jobs=-1
)
soft_voting_clf.fit(X_train, y_train)
y_pred_vote = soft_voting_clf.predict(X_test)
print("Soft Voting accuracy:", accuracy_score(y_test, y_pred_vote))
print(classification_report(y_test, y_pred_vote))

##########
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
final_estimator = LogisticRegression()
estimators=[
        ('rf',  best_rf),
        ('xgb', best_xgb),
        ('svc', best_svc)
    ]

stf_clf = StackingClassifier(estimators, 
                   final_estimator = final_estimator,  
                   cv=cv, stack_method='auto', 
                   n_jobs=None, passthrough=False, verbose=0)

stf_clf.fit(X_train, y_train)
y_pred_stf = stf_clf.predict(X_test)
print("Stacking Classifier accuracy:", accuracy_score(y_test, y_pred_stf))
print(classification_report(y_test, y_pred_stf))



from sklearn.metrics import f1_score
#ablation
models = {
    "KNN": best_knn,
    "RF": best_rf,
    "XGB": best_xgb,
    "SVC": best_svc,
    "SoftVote": soft_voting_clf,
    "Stacking": stf_clf,
}

results = []

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='binary')  # change to 'macro' if multi-class

    results.append({"Model": name, "Accuracy": acc, "F1": f1})

results_df = pd.DataFrame(results)
print(results_df)


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].bar(results_df["Model"], results_df["Accuracy"], color='skyblue')
axes[0].set_title("Accuracy by Model")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Score")
axes[0].set_xticklabels(results_df["Model"], rotation=45, ha='right')

axes[1].bar(results_df["Model"], results_df["F1"], color='salmon')
axes[1].set_title("F1-score by Model")
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(results_df["Model"], rotation=45, ha='right')

plt.tight_layout()
plt.show()

#vif after feature selection and before transformation
#pca after feature selection




base_models = {
    'rf':  best_rf,
    'xgb': best_xgb,
    'svc': best_svc
}

results = []

# all non‑empty subsets of base models (size 1, 2, 3)
for r in [1, 2, 3]:
    for combo in itertools.combinations(base_models.items(), r):
        name = "+".join([k for k, _ in combo])
        estimators = list(combo)

        meta = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000
        )

        stf_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=cv,              # your StratifiedKFold / LOO
            stack_method='auto',
            n_jobs=-1,
            passthrough=False,
            verbose=0
        )

        stf_clf.fit(X_train, y_train)
        y_pred = stf_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        print(f"\n=== {name} ===")
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))

        results.append({
            'Combo': name,
            'Num_Base': r,
            'Accuracy': acc,
            'F1': f1
        })

ablation_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print("\nAblation summary:")
print(ablation_df)



data = {
    'Combo': ['rf', 'svc', 'rf+svc', 'xgb+svc', 'xgb', 'rf+xgb', 'rf+xgb+svc'],
    'Num_Base': [1, 1, 2, 2, 1, 2, 3],
    'Accuracy': [0.818182, 0.818182, 0.818182, 0.818182, 0.787879, 0.787879, 0.787879],
    'F1':       [0.812500, 0.800000, 0.800000, 0.800000, 0.774194, 0.774194, 0.774194],
}

df = pd.DataFrame(data)

# Ensure order as in your table
df = df.reset_index(drop=True)

plt.figure(figsize=(8,4))
plt.plot(df['Combo'], df['Accuracy'], marker='o', label='Accuracy')
plt.plot(df['Combo'], df['F1'], marker='s', label='F1-score')
plt.ylim(0.75, 0.85)
plt.xlabel('Base-learner combination')
plt.ylabel('Score')
plt.title('Stacking Ablation: Accuracy & F1 per Combo')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
plt.plot(df['Num_Base'], df['Accuracy'], 'o-', label='Accuracy')
plt.plot(df['Num_Base'], df['F1'], 's--', label='F1-score')
plt.xlabel('Number of base learners')
plt.ylabel('Score')
plt.title('Performance vs number of base learners')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

####

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]   # smaller C = stronger L2
l2_results = []

for C in C_values:
    lf2 = LogisticRegression(
        penalty='l2',
        C=C,
        solver='lbfgs',
        max_iter=500
    )
    lf2.fit(X_train, y_train)
    y_pred = lf2.predict(X_test)
    l2_results.append({
        "Penalty": "L2",
        "C": C,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    })

l2_df = pd.DataFrame(l2_results)
print(l2_df)


penalty_results = []
lf2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=500)
lf2.fit(X_train, y_train)
y2 = lf2.predict(X_test)
penalty_results.append({
    "Model": "L2_C=1.0",
    "Accuracy": accuracy_score(y_test, y2),
    "F1": f1_score(y_test, y2)
})

# L1
lf1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=500)
lf1.fit(X_train, y_train)
y1 = lf1.predict(X_test)
penalty_results.append({
    "Model": "L1_C=1.0",
    "Accuracy": accuracy_score(y_test, y1),
    "F1": f1_score(y_test, y1)
})

# ElasticNet (mix of L1 + L2)
lfen = LogisticRegression(
    penalty='elasticnet',
    C=1.0,
    l1_ratio=0.5,          # 0.0=L2, 1.0=L1, between = mix
    solver='saga',
    max_iter=1000
)
lfen.fit(X_train, y_train)
yen = lfen.predict(X_test)
penalty_results.append({
    "Model": "ElasticNet_l1r=0.5",
    "Accuracy": accuracy_score(y_test, yen),
    "F1": f1_score(y_test, yen)
})

pen_df = pd.DataFrame(penalty_results)
print(pen_df)

plt.figure(figsize=(6,4))
plt.bar(pen_df["Model"], pen_df["Accuracy"], color=['C0','C1','C2'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Logistic Regression-penalty ablation")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()



feat_names = X.columns
rf_imp = best_rf.named_steps['clf'].feature_importances_ if hasattr(best_rf, 'named_steps') \
         else best_rf.feature_importances_
xgb_imp = best_xgb.named_steps['clf'].feature_importances_ if hasattr(best_xgb, 'named_steps') \
          else best_xgb.feature_importances_


svc_imp = permutation_importance(best_svc, X, y, n_repeats=20, random_state=42, n_jobs=-1).importances_mean
knn_imp = permutation_importance(best_knn, X, y, n_repeats=20, random_state=42, n_jobs=-1).importances_mean
rf_imp = permutation_importance(best_rf, X, y, n_repeats=20, random_state=42, n_jobs=-1).importances_mean
xgb_imp = permutation_importance(best_xgb, X, y, n_repeats=20, random_state=42, n_jobs=-1).importances_mean
svc_imp = permutation_importance(best_svc, X, y, n_repeats=20, random_state=42, n_jobs=-1).importances_mean
imp_df = pd.DataFrame({
    'Feature': feat_names,
    'KNN': knn_imp,
    'RF': rf_imp,
    'XGB': xgb_imp,
    'SVC': svc_imp
})
print(imp_df.head(10))



##feature grouping then modelling 

grp1_features = [ ##select accordingly
]
new_df = df_transformed[grp1_features]
print(new_df.head())
new_df.columns

X = new_df
y = df_transformed['label']
X = new_df[selected_features].copy()
X = X.fillna(X.median())
X.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train.shape, X_test.shape
cv = StratifiedKFold(
    n_splits=10,        # change to 10 if you want 10-fold CV
    shuffle=True,
    random_state=42
)
loo = LeaveOneOut()
knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
    
])
knn_param_grid = {
    
    'clf__n_neighbors': [3, 5, 7, 9, 11],
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan', 'minkowski']
}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid, cv=loo, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
print("KNN best params:", knn_grid.best_params_)
print("KNN best CV acc:", knn_grid.best_score_)
y_pred_knn = knn_grid.predict(X_test)
print("KNN test acc:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

############RF############
rf = RandomForestClassifier(random_state=42,class_weight='balanced')
rf_param_grid = {
    'criterion': ['entropy', 'gini'],        # tune/use entropy
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    
}

rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv,  scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("RF best params:", rf_grid.best_params_)
print("RF best CV acc:", rf_grid.best_score_)
y_pred_rf = rf_grid.predict(X_test)
print("RF test acc:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


#tree based feature importance
best_rf = rf_grid.best_estimator_

rf_importances = best_rf.feature_importances_
rf_imp_df = pd.DataFrame({
    'Feature': grp1_features,         
    'Importance': rf_importances
}).sort_values('Importance', ascending=False)

print(rf_imp_df)

plt.figure(figsize=(8, 6))
plt.barh(rf_imp_df['Feature'], rf_imp_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

############
rf2 = RandomForestClassifier(criterion='entropy', max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=10, n_estimators=500, random_state=42, class_weight= None)
rf2.fit(X_train, y_train)
y_pred_rf2 = rf2.predict(X_test)
print("RF2 test acc:", accuracy_score(y_test, y_pred_rf2))
print(classification_report(y_test, y_pred_rf2))

#######################3
from xgboost import XGBClassifier
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.5, 1]
}

xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

print("XGB best params:", xgb_grid.best_params_)
print("XGB best CV acc:", xgb_grid.best_score_)
y_pred_xgb = xgb_grid.predict(X_test)
print("XGB test acc:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))



best_xgb = xgb_grid.best_estimator_
xgb_importances = best_xgb.feature_importances_
xgb_imp_df = pd.DataFrame({
    'Feature': grp1_features,          # your selected feature list
    'Importance': xgb_importances
}).sort_values('Importance', ascending=False)

print(xgb_imp_df)

plt.figure(figsize=(8, 6))
plt.barh(xgb_imp_df['Feature'], xgb_imp_df['Importance'])
plt.gca().invert_yaxis()
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
 

##SVM

svc = SVC(probability=True, random_state=42)


# Random search space around your params
param_distributions = {
    'C': loguniform(1e-3, 1e3),
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5],          # used only if kernel='poly'
    'coef0': uniform(0, 1),         # for poly/sigmoid
    'class_weight': [None, 'balanced']
}
random_search_svc = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_distributions,
    n_iter=50,                # increase for more thorough search
    scoring='accuracy',
    cv=cv,                     # or your StratifiedKFold/LOO
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_svc.fit(X_train, y_train)
y_pred_svc = random_search_svc.predict(X_test)
print("SVC test acc:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
print("Best params:", random_search_svc.best_params_)
print("Best CV score:", random_search_svc.best_score_)

####
grp2_features = [###select accordingly 
]
new_df = df_transformed[grp2_features]
print(new_df.head())
new_df.columns

X = new_df
y = df_transformed['label']
X = new_df[selected_features].copy()
X = X.fillna(X.median())
X.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train.shape, X_test.shape
cv = StratifiedKFold(
    n_splits=10,        # change to 10 if you want 10-fold CV
    shuffle=True,
    random_state=42
)
loo = LeaveOneOut()
knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
    
])
knn_param_grid = {
    
    'clf__n_neighbors': [3, 5, 7, 9, 11],
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan', 'minkowski']
}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid, cv=loo, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
print("KNN best params:", knn_grid.best_params_)
print("KNN best CV acc:", knn_grid.best_score_)
y_pred_knn = knn_grid.predict(X_test)
print("KNN test acc:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

############RF############
rf = RandomForestClassifier(random_state=42,class_weight='balanced')
rf_param_grid = {
    'criterion': ['entropy', 'gini'],        # tune/use entropy
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    
}

rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv,  scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("RF best params:", rf_grid.best_params_)
print("RF best CV acc:", rf_grid.best_score_)
y_pred_rf = rf_grid.predict(X_test)
print("RF test acc:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
#######################3
from xgboost import XGBClassifier
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.5, 1]
}

xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

print("XGB best params:", xgb_grid.best_params_)
print("XGB best CV acc:", xgb_grid.best_score_)
y_pred_xgb = xgb_grid.predict(X_test)
print("XGB test acc:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))



best_xgb = xgb_grid.best_estimator_
xgb_importances = best_xgb.feature_importances_
xgb_imp_df = pd.DataFrame({
    'Feature': grp1_features,          # your selected feature list
    'Importance': xgb_importances
}).sort_values('Importance', ascending=False)

print(xgb_imp_df)

plt.figure(figsize=(8, 6))
plt.barh(xgb_imp_df['Feature'], xgb_imp_df['Importance'])
plt.gca().invert_yaxis()
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
 

##SVM

svc = SVC(probability=True, random_state=42)
# Random search space around your params
param_distributions = {
    'C': loguniform(1e-3, 1e3),
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5],          # used only if kernel='poly'
    'coef0': uniform(0, 1),         # for poly/sigmoid
    'class_weight': [None, 'balanced']
}
random_search_svc = RandomizedSearchCV(
    estimator=svc,
    param_distributions=param_distributions,
    n_iter=50,                # increase for more thorough search
    scoring='accuracy',
    cv=cv,                     # or your StratifiedKFold/LOO
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_svc.fit(X_train, y_train)
y_pred_svc = random_search_svc.predict(X_test)
print("SVC test acc:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
print("Best params:", random_search_svc.best_params_)
print("Best CV score:", random_search_svc.best_score_)
