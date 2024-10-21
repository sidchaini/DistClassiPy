---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.2
  nbformat: 4
  nbformat_minor: 5
---

::: {#8abc03bb-a8a8-42c4-89d7-e453093e5183 .cell .code execution_count="1" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:38.479307Z\",\"iopub.status.busy\":\"2024-05-13T23:09:38.476001Z\",\"iopub.status.idle\":\"2024-05-13T23:09:39.981814Z\",\"shell.execute_reply\":\"2024-05-13T23:09:39.981518Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:38.479280Z\"}"}
``` python
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.feature_selection import (
    SequentialFeatureSelector,
)
from mlxtend.evaluate import feature_importance_permutation
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import matplotlib.ticker as ticker
import os

os.chdir("../../")
from pathlib import Path
import json

import sys

sys.path.append("scripts")

import utils
import distclassipy as dcpy

cd = dcpy.Distance()
```
:::

::: {#3ec63bbb-e719-40e6-9adc-94b13fbf0994 .cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:39.982413Z\",\"iopub.status.busy\":\"2024-05-13T23:09:39.982258Z\",\"iopub.status.idle\":\"2024-05-13T23:09:39.985649Z\",\"shell.execute_reply\":\"2024-05-13T23:09:39.985389Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:39.982399Z\"}"}
``` python
with open("settings.txt") as f:
    settings_dict = json.load(f)
np.random.seed(settings_dict["seed_choice"])

classification_letter = "c"
classification_problem = settings_dict["classification_problem"][classification_letter]
classes_to_keep = settings_dict["classes_to_keep"][classification_letter]
results_subfolder = f"{classification_letter}. {classification_problem}"
sns_dict = settings_dict["sns_dict"]

sns.set_theme(**sns_dict)
```
:::

::: {#90b0c36b-9011-4bd0-9db8-e9a1a67c2bb0 .cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:39.986136Z\",\"iopub.status.busy\":\"2024-05-13T23:09:39.986054Z\",\"iopub.status.idle\":\"2024-05-13T23:09:40.150232Z\",\"shell.execute_reply\":\"2024-05-13T23:09:40.149983Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:39.986127Z\"}"}
``` python
check_estimator(dcpy.DistanceMetricClassifier())  # passes
```
:::

::: {#b8d4ef96-50ea-46e8-844b-b195caccf7e9 .cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:40.150723Z\",\"iopub.status.busy\":\"2024-05-13T23:09:40.150633Z\",\"iopub.status.idle\":\"2024-05-13T23:09:40.219267Z\",\"shell.execute_reply\":\"2024-05-13T23:09:40.219008Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:40.150714Z\"}"}
``` python
# Load Data
X_df_FULL = pd.read_csv("data/X_df.csv", index_col=0)
y_df_FULL = pd.read_csv("data/y_df.csv", index_col=0)
```
:::

:::: {#2b043e57-90ae-45ba-a3f2-37db02771085 .cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:40.219916Z\",\"iopub.status.busy\":\"2024-05-13T23:09:40.219783Z\",\"iopub.status.idle\":\"2024-05-13T23:09:40.225898Z\",\"shell.execute_reply\":\"2024-05-13T23:09:40.224884Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:40.219907Z\"}"}
``` python
# Remove manually selected 'bad' features
with open(os.path.join("results", results_subfolder, "drop_features.txt")) as f:
    bad_features = json.load(f)  # manually selected

X_df_FULL = X_df_FULL.drop(bad_features, axis=1)

print(X_df_FULL.shape[1])
```

::: {.output .stream .stdout}
    31
:::
::::

::: {#985db29c-1b60-47de-aab9-e2f0cc96a9a3 .cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:40.230066Z\",\"iopub.status.busy\":\"2024-05-13T23:09:40.229769Z\",\"iopub.status.idle\":\"2024-05-13T23:09:40.234651Z\",\"shell.execute_reply\":\"2024-05-13T23:09:40.234240Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:40.230055Z\"}"}
``` python
# Keep only current classes
cl_keep_str = "_".join(classes_to_keep)

y_df = y_df_FULL[y_df_FULL["class"].isin(classes_to_keep)]
X_df = X_df_FULL.loc[y_df.index]
X = X_df.to_numpy()
y = y_df.to_numpy().ravel()
```
:::

::: {#39dac2d6-28ff-41da-af10-aa2bdfeee1df .cell .code execution_count="7" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:40.235476Z\",\"iopub.status.busy\":\"2024-05-13T23:09:40.235163Z\",\"iopub.status.idle\":\"2024-05-13T23:09:40.239152Z\",\"shell.execute_reply\":\"2024-05-13T23:09:40.238533Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:40.235465Z\"}" tags="[]"}
``` python
all_metrics = settings_dict["all_metrics"]
scoring = "f1_macro"
```
:::

::: {#2ed3ad68-1b0b-4da1-b6fc-7e26a6d0d51e .cell .code execution_count="8" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:40.240669Z\",\"iopub.status.busy\":\"2024-05-13T23:09:40.240468Z\",\"iopub.status.idle\":\"2024-05-13T23:09:40.245578Z\",\"shell.execute_reply\":\"2024-05-13T23:09:40.244172Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:40.240658Z\"}" tags="[]"}
``` python
from IPython.display import IFrame
```
:::

::: {#cc3c599d-e6af-42db-b944-ca78b45222ed .cell .code execution_count="9" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:40.246945Z\",\"iopub.status.busy\":\"2024-05-13T23:09:40.246564Z\",\"iopub.status.idle\":\"2024-05-13T23:09:40.253932Z\",\"shell.execute_reply\":\"2024-05-13T23:09:40.252320Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:40.246770Z\"}" tags="[]"}
``` python
with open(os.path.join("results", results_subfolder, "best_common_features.txt")) as f:
    best_common_features = json.load(f)
```
:::

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {#6cee789c-03e5-485e-a5a3-784d43d6c72f .cell .code execution_count="10" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:40.255776Z\",\"iopub.status.busy\":\"2024-05-13T23:09:40.255499Z\",\"iopub.status.idle\":\"2024-05-13T23:09:56.031123Z\",\"shell.execute_reply\":\"2024-05-13T23:09:56.030738Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:40.255748Z\"}" scrolled="true" tags="[]"}
``` python
# for metric in tqdm([cd.wave_hedges]): EXAMINE wave_hedges, clark, add_chisq, maryland bridge

bestfeat_robust = []
common_best_feat = best_common_features[0]

for metric in tqdm(all_metrics):
    # for metric in tqdm([cd.clark, 'canberra', cd.hellinger]):
    metric_str = utils.get_metric_name(metric)
    locpath = os.path.join("results", results_subfolder, "distclassipy", metric_str)
    print("*" * 20, metric_str, "*" * 20)
    lcdc = dcpy.DistanceMetricClassifier(
        metric=metric,
        scale=True,
        central_stat=settings_dict["central_stat"],
        dispersion_stat=settings_dict["dispersion_stat"],
        calculate_kde=False,
        calculate_1d_dist=False,
    )

    # Load best min features within 1 std of best score.
    sfs_df = pd.read_csv(os.path.join(locpath, "sfs_allfeatures.csv"), index_col=0)
    feats_idx, feats = utils.load_best_features(sfs_df)

    all_feats = X_df.columns

    # show SFS plot from before for this metric
    filepath = os.path.join("../../", locpath, "sfs_allfeatures_plot_marked.pdf")
    display(IFrame(filepath, width=700, height=500))

    # Calculate the score for each quantile
    newfeats = []
    feat_quantile_scores = []

    # print("Total test score of all quantiles together:")

    # pbar = tqdm(feats, leave=False) # Loop through "best feats"
    pbar = tqdm(best_common_features, leave=False)  # Loop through best_common_features
    for feat in pbar:
        pbar.set_description(f"Feature: {feat}")
        try:
            quantiles = pd.qcut(X_df.loc[:, feat], q=4)  # q is number of splits
        except ValueError as ve:
            print(f"{feat}: {ve}. Continuing.")
            continue

        X_df_subset = X_df.loc[
            :, feats
        ]  # X_df with just the features important for this metric
        # X_df_subset["quantile"] = quantiles

        X = X_df_subset.to_numpy()
        # X_train, X_test, y_train, y_test = train_test_split(X_df_subset,y, test_size=0.33, stratify=[feat])
        X_train, X_test, y_train, y_test = train_test_split(
            X_df_subset, y_df, test_size=0.33, stratify=quantiles, random_state=44
        )

        lcdc.fit(X_train, y_train.to_numpy().ravel())
        acc_tot = lcdc.score(X_test, y_test)
        # y_preds_tot = lcdc.predict(X_test)
        # f1_tot = f1_score(y_preds_tot, y_test, average="macro")
        print(f"\t{feat} feature: {acc_tot:.2f}")
        # print(f"\t{feat} feature: {f1_tot:.2f}")
        grouped = X_test.groupby(quantiles)

        quantile_scores = []
        for i, (lims, subdf) in enumerate(grouped):
            y_pred = lcdc.predict(subdf.to_numpy())
            y_true = y_test.loc[subdf.index]
            # print(y_true["class"].value_counts())
            # f1 = f1_score(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred)

            # quantile_scores.append(f1)
            quantile_scores.append(acc)
        feat_quantile_scores.append(quantile_scores)
        newfeats.append(feat)

    feat_quantile_scores = np.array(feat_quantile_scores) * 100  # Change to percentage

    feat_quantile_scores_df = pd.DataFrame(
        data=feat_quantile_scores,
        index=newfeats,
        columns=[f"Quantile {i+1}" for i in range(4)],
    )
    # sorted by best-wors feats by default

    # CHANGE AXES FIG RATIO

    fig, ax = plt.subplots()
    sns.heatmap(
        feat_quantile_scores_df,
        ax=ax,
        annot=True,
        fmt=".0f",
        vmin=65,
        vmax=100,
        cmap="Blues",
        cbar_kws={"label": "Accuracy (%)"},
    )

    ax.set_title(f"{metric_str.title()} metric")
    ax.set_ylabel("Feature Name")

    # Embolden (or color red) the selected labels on the y axis
    ytick_labels = ax.get_yticklabels()
    # label_colors = [sns.color_palette()[3] if f in feats else 'black' for f in pbar]
    label_fontweights = ["bold" if f in feats else None for f in pbar]

    # for tick_label, color in zip(ytick_labels, label_colors):
    #     tick_label.set_color(color)
    for tick_label, fw in zip(ytick_labels, label_fontweights):
        tick_label.set_fontweight(fw)

    plt.savefig(os.path.join(locpath, "robustness_plot.pdf"), bbox_inches="tight")
    plt.show()

    sr = feat_quantile_scores_df.loc[common_best_feat]
    sr.name = metric_str
    bestfeat_robust.append(sr)
```

::: {.output .display_data}
``` json
{"model_id":"2924a7dae2ac47449eb706a1dd96f593","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    ******************** Euclidean ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Euclidean/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.79
    	Harmonics_phase_2_r feature: 0.77
    	IAR_phi_r feature: 0.77
    	GP_DRW_tau_r feature: 0.79
    	MHPS_low_r feature: 0.78
    	Rcs_r feature: 0.81
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	MHPS_ratio_r feature: 0.78
    	Harmonics_mag_1_r feature: 0.80
    	Psi_eta_r feature: 0.80
    	Harmonics_phase_5_r feature: 0.80
    	Amplitude_r feature: 0.82
    	Harmonics_mag_7_r feature: 0.74
    	Beyond1Std_r feature: 0.79
    	Eta_e_r feature: 0.81
    	Gskew_r feature: 0.77
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](2519404982312788545440527a15576bfc1d6440.png)
:::

::: {.output .stream .stdout}
    ******************** Braycurtis ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Braycurtis/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.91
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.91
    	GP_DRW_tau_r feature: 0.90
    	MHPS_low_r feature: 0.89
    	Rcs_r feature: 0.90
    	MHPS_ratio_r feature: 0.91
    	Harmonics_mag_1_r feature: 0.91
    	Psi_eta_r feature: 0.90
    	Harmonics_phase_5_r feature: 0.91
    	Amplitude_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Harmonics_mag_7_r feature: 0.89
    	Beyond1Std_r feature: 0.90
    	Eta_e_r feature: 0.90
    	Gskew_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](817bbe234f45c48c171de8ae3490638a00c02ac4.png)
:::

::: {.output .stream .stdout}
    ******************** Canberra ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Canberra/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.90
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.91
    	GP_DRW_tau_r feature: 0.90
    	MHPS_low_r feature: 0.91
    	Rcs_r feature: 0.91
    	MHPS_ratio_r feature: 0.91
    	Harmonics_mag_1_r feature: 0.91
    	Psi_eta_r feature: 0.91
    	Harmonics_phase_5_r feature: 0.92
    	Amplitude_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Harmonics_mag_7_r feature: 0.90
    	Beyond1Std_r feature: 0.91
    	Eta_e_r feature: 0.91
    	Gskew_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](dfb11a78413b79817ff4bd78bfe16d6a6bc144a2.png)
:::

::: {.output .stream .stdout}
    ******************** Cityblock ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Cityblock/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.83
    	Harmonics_phase_2_r feature: 0.82
    	IAR_phi_r feature: 0.83
    	GP_DRW_tau_r feature: 0.85
    	MHPS_low_r feature: 0.85
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Rcs_r feature: 0.86
    	MHPS_ratio_r feature: 0.83
    	Harmonics_mag_1_r feature: 0.87
    	Psi_eta_r feature: 0.87
    	Harmonics_phase_5_r feature: 0.86
    	Amplitude_r feature: 0.84
    	Harmonics_mag_7_r feature: 0.78
    	Beyond1Std_r feature: 0.84
    	Eta_e_r feature: 0.81
    	Gskew_r feature: 0.82
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](645ba224512c8c0606512a7e50e155807fa448c3.png)
:::

::: {.output .stream .stdout}
    ******************** Chebyshev ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Chebyshev/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.75
    	Harmonics_phase_2_r feature: 0.77
    	IAR_phi_r feature: 0.76
    	GP_DRW_tau_r feature: 0.74
    	MHPS_low_r feature: 0.76
    	Rcs_r feature: 0.76
    	MHPS_ratio_r feature: 0.75
    	Harmonics_mag_1_r feature: 0.74
    	Psi_eta_r feature: 0.76
    	Harmonics_phase_5_r feature: 0.79
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Amplitude_r feature: 0.74
    	Harmonics_mag_7_r feature: 0.73
    	Beyond1Std_r feature: 0.75
    	Eta_e_r feature: 0.75
    	Gskew_r feature: 0.74
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](d7aac93d20dc1039e8f047777d84ff9fbb29ad7f.png)
:::

::: {.output .stream .stdout}
    ******************** Clark ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Clark/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.90
    	Harmonics_phase_2_r feature: 0.92
    	IAR_phi_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	GP_DRW_tau_r feature: 0.91
    	MHPS_low_r feature: 0.91
    	Rcs_r feature: 0.91
    	MHPS_ratio_r feature: 0.90
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Harmonics_mag_1_r feature: 0.91
    	Psi_eta_r feature: 0.93
    	Harmonics_phase_5_r feature: 0.93
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Amplitude_r feature: 0.92
    	Harmonics_mag_7_r feature: 0.90
    	Beyond1Std_r feature: 0.91
    	Eta_e_r feature: 0.93
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Gskew_r feature: 0.93
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](96a07e211bf8573a38ee3e1ca0f2a76b65c551bd.png)
:::

::: {.output .stream .stdout}
    ******************** Correlation ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Correlation/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"1a606ecfa7c04667bfc525710ad79232","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.80
    	Harmonics_phase_2_r feature: 0.82
    	IAR_phi_r feature: 0.81
    	GP_DRW_tau_r feature: 0.80
    	MHPS_low_r feature: 0.81
    	Rcs_r feature: 0.82
    	MHPS_ratio_r feature: 0.83
    	Harmonics_mag_1_r feature: 0.80
    	Psi_eta_r feature: 0.82
    	Harmonics_phase_5_r feature: 0.82
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Amplitude_r feature: 0.82
    	Harmonics_mag_7_r feature: 0.82
    	Beyond1Std_r feature: 0.82
    	Eta_e_r feature: 0.81
    	Gskew_r feature: 0.81
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](1611f3e2b5d94fe9fc6370ebf395d024b5951a72.png)
:::

::: {.output .stream .stdout}
    ******************** Cosine ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Cosine/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"bee4611a6c9d48c3a972d040366c25f0","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.83
    	Harmonics_phase_2_r feature: 0.83
    	IAR_phi_r feature: 0.84
    	GP_DRW_tau_r feature: 0.83
    	MHPS_low_r feature: 0.84
    	Rcs_r feature: 0.85
    	MHPS_ratio_r feature: 0.84
    	Harmonics_mag_1_r feature: 0.85
    	Psi_eta_r feature: 0.86
    	Harmonics_phase_5_r feature: 0.83
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Amplitude_r feature: 0.87
    	Harmonics_mag_7_r feature: 0.84
    	Beyond1Std_r feature: 0.83
    	Eta_e_r feature: 0.84
    	Gskew_r feature: 0.83
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](bd6b61e5bfdc19b49b8dd9f2b417ba19cdd515e3.png)
:::

::: {.output .stream .stdout}
    ******************** Hellinger ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Hellinger/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"6e58947accbe4cfab5560e77f96fcb4c","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.91
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.92
    	GP_DRW_tau_r feature: 0.91
    	MHPS_low_r feature: 0.91
    	Rcs_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	MHPS_ratio_r feature: 0.91
    	Harmonics_mag_1_r feature: 0.92
    	Psi_eta_r feature: 0.92
    	Harmonics_phase_5_r feature: 0.92
    	Amplitude_r feature: 0.91
    	Harmonics_mag_7_r feature: 0.90
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Beyond1Std_r feature: 0.91
    	Eta_e_r feature: 0.92
    	Gskew_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](eee389f1f7a60ae8005dffa54517a81352e54cba.png)
:::

::: {.output .stream .stdout}
    ******************** Jaccard ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Jaccard/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"6f6bb988ab2744369bfe4bb091d01f68","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.87
    	Harmonics_phase_2_r feature: 0.89
    	IAR_phi_r feature: 0.88
    	GP_DRW_tau_r feature: 0.89
    	MHPS_low_r feature: 0.87
    	Rcs_r feature: 0.88
    	MHPS_ratio_r feature: 0.89
    	Harmonics_mag_1_r feature: 0.88
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Psi_eta_r feature: 0.90
    	Harmonics_phase_5_r feature: 0.89
    	Amplitude_r feature: 0.89
    	Harmonics_mag_7_r feature: 0.88
    	Beyond1Std_r feature: 0.89
    	Eta_e_r feature: 0.89
    	Gskew_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](0756ac9a5f9d25bc5efda8c3533f3a42d9abba72.png)
:::

::: {.output .stream .stdout}
    ******************** Lorentzian ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Lorentzian/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"007155a25e5642b3a25e5762f8bc8425","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.86
    	Harmonics_phase_2_r feature: 0.85
    	IAR_phi_r feature: 0.86
    	GP_DRW_tau_r feature: 0.89
    	MHPS_low_r feature: 0.86
    	Rcs_r feature: 0.88
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	MHPS_ratio_r feature: 0.86
    	Harmonics_mag_1_r feature: 0.89
    	Psi_eta_r feature: 0.89
    	Harmonics_phase_5_r feature: 0.89
    	Amplitude_r feature: 0.86
    	Harmonics_mag_7_r feature: 0.82
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Beyond1Std_r feature: 0.87
    	Eta_e_r feature: 0.84
    	Gskew_r feature: 0.85
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](56613266eb41bfee7dec2746a420676319204c38.png)
:::

::: {.output .stream .stdout}
    ******************** Marylandbridge ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Marylandbridge/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"3fab3ff163af43e8b4d851984d163cb4","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.74
    	Harmonics_phase_2_r feature: 0.73
    	IAR_phi_r feature: 0.74
    	GP_DRW_tau_r feature: 0.73
    	MHPS_low_r feature: 0.73
    	Rcs_r feature: 0.75
    	MHPS_ratio_r feature: 0.74
    	Harmonics_mag_1_r feature: 0.72
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Psi_eta_r feature: 0.73
    	Harmonics_phase_5_r feature: 0.77
    	Amplitude_r feature: 0.77
    	Harmonics_mag_7_r feature: 0.75
    	Beyond1Std_r feature: 0.77
    	Eta_e_r feature: 0.76
    	Gskew_r feature: 0.75
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](b72ffd6e7b18353fd2426a88f2baf6cbb1b592e9.png)
:::

::: {.output .stream .stdout}
    ******************** Meehl ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Meehl/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"84e0a641efa84ce1b687638c0b9b83b6","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.79
    	Harmonics_phase_2_r feature: 0.80
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	IAR_phi_r feature: 0.79
    	GP_DRW_tau_r feature: 0.80
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	MHPS_low_r feature: 0.78
    	Rcs_r feature: 0.80
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	MHPS_ratio_r feature: 0.78
    	Harmonics_mag_1_r feature: 0.79
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Psi_eta_r feature: 0.79
    	Harmonics_phase_5_r feature: 0.80
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Amplitude_r feature: 0.80
    	Harmonics_mag_7_r feature: 0.76
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Beyond1Std_r feature: 0.78
    	Eta_e_r feature: 0.82
    	Gskew_r feature: 0.75
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](73dddc4a54b99d4e8e951fed02376d94626bc0d2.png)
:::

::: {.output .stream .stdout}
    ******************** Motyka ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Motyka/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"3577cb3c19b34b43a9fa20519841e889","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.91
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.91
    	GP_DRW_tau_r feature: 0.90
    	MHPS_low_r feature: 0.89
    	Rcs_r feature: 0.90
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	MHPS_ratio_r feature: 0.91
    	Harmonics_mag_1_r feature: 0.91
    	Psi_eta_r feature: 0.90
    	Harmonics_phase_5_r feature: 0.91
    	Amplitude_r feature: 0.91
    	Harmonics_mag_7_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Beyond1Std_r feature: 0.90
    	Eta_e_r feature: 0.90
    	Gskew_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](5a531d43c3156ac206ae27465d0fbff988775ea0.png)
:::

::: {.output .stream .stdout}
    ******************** Soergel ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Soergel/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"aff65e8b2df44333a94bfc5a74d31464","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.91
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.91
    	GP_DRW_tau_r feature: 0.90
    	MHPS_low_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Rcs_r feature: 0.90
    	MHPS_ratio_r feature: 0.91
    	Harmonics_mag_1_r feature: 0.91
    	Psi_eta_r feature: 0.90
    	Harmonics_phase_5_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Amplitude_r feature: 0.91
    	Harmonics_mag_7_r feature: 0.89
    	Beyond1Std_r feature: 0.90
    	Eta_e_r feature: 0.90
    	Gskew_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](19e2a909964c6d28b78079fb7883d9d0b2f94d77.png)
:::

::: {.output .stream .stdout}
    ******************** Wave_Hedges ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Wave_Hedges/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"7078503af0cf44fb9953ab41781a2312","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.90
    	Harmonics_phase_2_r feature: 0.90
    	IAR_phi_r feature: 0.92
    	GP_DRW_tau_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}

    	MHPS_low_r feature: 0.90
    	Rcs_r feature: 0.91
    	MHPS_ratio_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Harmonics_mag_1_r feature: 0.90
    	Psi_eta_r feature: 0.91
    	Harmonics_phase_5_r feature: 0.91
    	Amplitude_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Harmonics_mag_7_r feature: 0.90
    	Beyond1Std_r feature: 0.91
    	Eta_e_r feature: 0.92
    	Gskew_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](cd8dc1257dc62e87417586c85ab728ed32900cff.png)
:::

::: {.output .stream .stdout}
    ******************** Kulczynski ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Kulczynski/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"c649789c07fb4420a65417b4ad7a43d3","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.91
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.91
    	GP_DRW_tau_r feature: 0.90
    	MHPS_low_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Rcs_r feature: 0.90
    	MHPS_ratio_r feature: 0.91
    	Harmonics_mag_1_r feature: 0.91
    	Psi_eta_r feature: 0.90
    	Harmonics_phase_5_r feature: 0.91
    	Amplitude_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}

    	Harmonics_mag_7_r feature: 0.89
    	Beyond1Std_r feature: 0.90
    	Eta_e_r feature: 0.90
    	Gskew_r feature: 0.89
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](b8afec1132e9c5e6eb1d358f7c4c66d3cc5de123.png)
:::

::: {.output .stream .stdout}
    ******************** Add_Chisq ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="../../results/c. multiclass/distclassipy/Add_Chisq/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"d66a3b77965347a7b9c94479d9af5538","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.92
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.91
    	GP_DRW_tau_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	MHPS_low_r feature: 0.91
    	Rcs_r feature: 0.92
    	MHPS_ratio_r feature: 0.91
    	Harmonics_mag_1_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Psi_eta_r feature: 0.92
    	Harmonics_phase_5_r feature: 0.92
    	Amplitude_r feature: 0.92
    	Harmonics_mag_7_r feature: 0.91
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .stream .stdout}
    	Beyond1Std_r feature: 0.91
    	Eta_e_r feature: 0.92
    	Gskew_r feature: 0.92
:::

::: {.output .stream .stderr}
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
    /var/folders/2d/ht34x6mn7hx9d1sv4g_j8nfr0000gn/T/ipykernel_36321/3380787777.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      grouped = X_test.groupby(quantiles)
:::

::: {.output .display_data}
![](d4e45d33a78e8ac27a2df5c66cfa9e963c4a722f.png)
:::

::: {.output .display_data}
![](9dab521a8c94e29f2b60ed073efd8374dd09ac59.png)
:::

::: {.output .stream .stdout}
    ******************** Add_Chisq ********************
:::

::: {.output .display_data}
```{=html}

        <iframe
            width="700"
            height="500"
            src="results/c. multiclass/distclassipy/Add_Chisq/sfs_allfeatures_plot_marked.pdf"
            frameborder="0"
            allowfullscreen
            
        ></iframe>
        
```
:::

::: {.output .display_data}
``` json
{"model_id":"b188da273dcf4c4086e0730cbed9002a","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    	Period_band_r feature: 0.92
    	Harmonics_phase_2_r feature: 0.91
    	IAR_phi_r feature: 0.91
    	GP_DRW_tau_r feature: 0.92
:::

::: {.output .stream .stdout}

    	MHPS_low_r feature: 0.91
    	Rcs_r feature: 0.92
    	MHPS_ratio_r feature: 0.91
:::

::: {.output .stream .stdout}
    	Harmonics_mag_1_r feature: 0.92
    	Psi_eta_r feature: 0.92
    	Harmonics_phase_5_r feature: 0.92
    	Amplitude_r feature: 0.92
:::

::: {.output .stream .stdout}
    	Harmonics_mag_7_r feature: 0.91
    	Beyond1Std_r feature: 0.91
    	Eta_e_r feature: 0.92
    	Gskew_r feature: 0.92
:::

::: {.output .display_data}
![](5874d78df4c845a013539d2b5d6de4dd8923747e.png)
:::

::: {.output .stream .stdout}
    	Psi_eta_r feature: 0.92
    	Harmonics_phase_5_r feature: 0.92
    	Amplitude_r feature: 0.92
    	Harmonics_mag_7_r feature: 0.91
:::

::: {.output .stream .stdout}
    	Beyond1Std_r feature: 0.91
    	Eta_e_r feature: 0.92
    	Gskew_r feature: 0.92
:::

::: {.output .display_data}
![](08f6c0273c0a73b24f811bdbd2c3647d74c48ba1.png)
:::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

::: {#502091b5-0d7f-4609-a7e2-e9afc8a3a8ce .cell .code execution_count="11" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:56.031820Z\",\"iopub.status.busy\":\"2024-05-13T23:09:56.031683Z\",\"iopub.status.idle\":\"2024-05-13T23:09:56.035060Z\",\"shell.execute_reply\":\"2024-05-13T23:09:56.034727Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:56.031807Z\"}" tags="[]"}
``` python
bestfeat_robust = pd.DataFrame(bestfeat_robust)
bestfeat_robust.index.name = common_best_feat
bestfeat_robust.index = bestfeat_robust.index.str.title()
```
:::

::: {#3ba8bde3-3c90-4a98-b58d-937f05df97b4 .cell .code execution_count="12" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:56.035734Z\",\"iopub.status.busy\":\"2024-05-13T23:09:56.035614Z\",\"iopub.status.idle\":\"2024-05-13T23:09:56.041029Z\",\"shell.execute_reply\":\"2024-05-13T23:09:56.039323Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:56.035720Z\"}" tags="[]"}
``` python
columns = [f"Quantile {i+1}" for i in range(4)]
```
:::

:::: {#b1fd89d2-1762-41b4-a500-5952a9852c59 .cell .code execution_count="13" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:56.049571Z\",\"iopub.status.busy\":\"2024-05-13T23:09:56.047773Z\",\"iopub.status.idle\":\"2024-05-13T23:09:56.058257Z\",\"shell.execute_reply\":\"2024-05-13T23:09:56.056440Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:56.049530Z\"}"}
``` python
locpath
```

::: {.output .execute_result execution_count="13"}
    'results/c. multiclass/distclassipy/Add_Chisq'
:::
::::

:::: {#a32057e6-caf6-4c06-a1af-22745dfe1e8f .cell .code execution_count="14" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:56.064686Z\",\"iopub.status.busy\":\"2024-05-13T23:09:56.061809Z\",\"iopub.status.idle\":\"2024-05-13T23:09:56.351862Z\",\"shell.execute_reply\":\"2024-05-13T23:09:56.351459Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:56.064662Z\"}" tags="[]"}
``` python
fig, ax = plt.subplots()
sns.heatmap(
    bestfeat_robust,
    ax=ax,
    annot=True,
    fmt=".0f",
    vmin=65,
    vmax=100,
    cmap="Blues",
    cbar_kws={"label": "Accuracy (%)"},
)

ax.set_title(f"Quantiles for {common_best_feat}")
ax.set_ylabel("Metric")

plt.savefig(
    os.path.join(
        "results", results_subfolder, "distclassipy", "robustness_bestfeat.pdf"
    ),
    bbox_inches="tight",
)
plt.show()
```

::: {.output .display_data}
![](177f109d8c026a809873903037e03aaabf28730a.png)
:::
::::

::: {#9cf9f252-2dbf-49d0-8891-aeb0ab15895b .cell .code execution_count="15" execution="{\"iopub.execute_input\":\"2024-05-13T23:09:56.352526Z\",\"iopub.status.busy\":\"2024-05-13T23:09:56.352420Z\",\"iopub.status.idle\":\"2024-05-13T23:09:56.362768Z\",\"shell.execute_reply\":\"2024-05-13T23:09:56.358337Z\",\"shell.execute_reply.started\":\"2024-05-13T23:09:56.352517Z\"}" tags="[]"}
``` python
# fig, ax = plt.subplots()
# sns.heatmap(bestfeat_robust, ax=ax,
#             annot=True, fmt='.0f',
#             # vmin=80,
#             vmax=100,
#             cmap="Blues",
#             cbar_kws={'label': 'Accuracy (%)'})

# ax.set_title(f"Quantiles for {common_best_feat}")
# ax.set_ylabel("Metric")

# plt.savefig(f"results/robustness_bestfeat.svg", bbox_inches = 'tight')
# plt.show()
```
:::

::: {#6b7d3ad0-98fe-4f3a-9663-4af1a674b0e0 .cell .markdown}
In the robustness plot, I added all features on y axis, with the
relevant ones for each metric coloured red. I also created a robustness
plot for different features
:::
