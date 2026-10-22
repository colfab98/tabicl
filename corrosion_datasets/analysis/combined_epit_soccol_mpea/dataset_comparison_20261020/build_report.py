"""Build a self-contained HTML report from the saved, reproducible diagnostics."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reproduce import OUT, TARGET


def table(frame, digits=3):
    return frame.to_html(index=False,border=0,float_format=lambda v:f"{v:.{digits}f}",na_rep="—",escape=True)


def image(name, caption):
    encoded=base64.b64encode((OUT/name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def main():
    summary=pd.read_csv(OUT/"dataset_summary.csv").set_index("dataset")
    features=pd.read_csv(OUT/"feature_distributions.csv")
    missing=features.pivot(index="feature",columns="dataset",values="missing_fraction")
    profiles=json.loads((OUT/"prior_profiles.json").read_text())
    q=pd.read_csv(OUT/"mpea_row_descriptors.csv")
    metrics=pd.read_csv(OUT/"model_diagnostics/fold_metrics.csv")
    groups=pd.read_csv(OUT/"catboost_feature_group_metrics.csv")
    assert len(metrics)==30 and len(groups)==28
    model_means=metrics.groupby(["model","variant"])[["spearman","rmse"]].mean()
    group_means=groups.groupby(["dataset","feature_group"])[["spearman","rmse"]].mean()
    relation=pd.read_csv(OUT/"target_relationships.csv")
    bootstrap=pd.read_csv(OUT/"paired_study_bootstrap.csv")
    within=pd.read_csv(OUT/"model_within_study_ranks.csv")
    subgroup=pd.read_csv(OUT/"model_subgroup_metrics.csv")
    prior_correlation=pd.read_csv(OUT/"prior_composition_correlation_shift.csv")
    names={"epit_shared_v2":"EPIT specialist","soccol_shared_v2":"Soccol specialist",
           "combined_corrpfn_v2":"COMB v2"}
    overview=[]
    for source in ["EPIT","Soccol","MPEA"]:
        s=summary.loc[source]
        overview.append({
            "Dataset":source,"Rows":int(s.rows),"Study IDs":int(s.studies),
            "Fe ≥50 wt% (%)":100*s.Fe_ge50_fraction,
            "Median largest element (wt%)":s.max_reported_element_median_wt_pct,
            "Median elements ≥5 wt%":s.n_elements_ge5wt_median,
            "Target median (mV)":s.target_median,"Target IQR (mV)":s.target_q75-s.target_q25,
        })
    diagnostic=[]
    for label in names:
        a=model_means.loc[(label,"original")]
        b=model_means.loc[(label,"hide_soccol_unsupported_composition")]
        diagnostic.append({"Model":names[label],"Original mean rho":a.spearman,
                           "11 columns hidden: mean rho":b.spearman,
                           "Original mean RMSE (mV)":a.rmse,"Hidden: mean RMSE (mV)":b.rmse})
    cb=[]
    for source in ["EPIT","Soccol","MPEA"]:
        cb.append({"Dataset":source,**{
            name:group_means.loc[(source,key),"spearman"]
            for name,key in [("All inputs","all_features"),("Composition only","composition_only"),
                             ("Composition + chloride","composition_and_chloride"),
                             ("Non-composition only","noncomposition_only")]
        }})
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout="constrained")
    for i,(key,label,color) in enumerate([
        ("all_features","All inputs","#2879b5"),("composition_only","Composition only","#d76c2b"),
        ("noncomposition_only","Non-composition only","#23856c")]):
        vals=[group_means.loc[(s,key),"spearman"] for s in ["EPIT","Soccol","MPEA"]]
        ax.bar(np.arange(3)+(i-1)*.24,vals,width=.22,label=label,color=color)
    ax.set_xticks(range(3),["EPIT: original final test","Soccol: original final test","MPEA: five outer folds"])
    ax.set_ylim(0,.85);ax.set_ylabel("Spearman");ax.legend(fontsize=9)
    ax.set_title("Fixed CatBoost probe: which recorded inputs support prediction?")
    fig.savefig(OUT/"feature_group_probe.png",dpi=170);plt.close(fig)
    rt=relation[relation.feature.isin(["Cr","Mo","Cu","Ti","Nb","CP_Cl"])].pivot(
        index="feature",columns="dataset",values="spearman"
    ).reindex(["Cr","Mo","Cu","Ti","Nb","CP_Cl"])[["EPIT","Soccol","MPEA"]].reset_index()
    subset=subgroup[subgroup["subset"].isin([
        "all","NaCl_seawater","chloride_0p6M","no_elements_omitted_from_shared_schema"
    ])]
    subset=subset.pivot(index=["subset","rows","studies"],columns="model",values="pooled_prediction_spearman").reset_index()
    subset=subset.rename(columns=names)
    ranks=within[["model","pooled_rho","study_and_fold_centered_global_rank_correlation",
                  "study_and_fold_centered_within_group_rank_correlation"]].copy()
    ranks["model"]=ranks.model.map(names)
    ranks=ranks.rename(columns={"model":"Model","pooled_rho":"Pooled rho",
        "study_and_fold_centered_global_rank_correlation":"Study/fold-centered global-rank correlation",
        "study_and_fold_centered_within_group_rank_correlation":"Centered within-group rank correlation"})
    env=pd.read_csv(OUT/"mpea_environment_targets.csv")
    env=env[["environment_family","rows","studies","target_median","target_minimum","target_maximum"]]
    body=f"""
<header><p class="eyebrow">Post-hoc research analysis · 20 October 2026</p>
<h1>Why does EPIT transfer to MPEA better than Soccol?</h1>
<p class="lead">The strongest explanation is <strong>over-specialization of the Soccol training recipe to its source task</strong>:
steel-dominated composition, many measured environmental/procedure variables, and strong reliance on fitted analytical rules.
MPEA requires learning new composition relationships from sparse context. EPIT's recipe retains substantially more generic learning.
The analysis supports this explanation, but does not isolate one training hyperparameter as the sole cause.</p></header>

<section><h2>What the evidence establishes</h2>
<ul>
<li><strong>Different available information:</strong> MPEA exposes an average 17.72 of EPIT's 28 native inputs,
but only 9.74 of Soccol's 37. Temperature is recorded in 5/267 rows; pH and scan rate are absent throughout.</li>
<li><strong>Different reliance on the source task:</strong> both frozen formula mixtures have weak compatibility with MPEA
(ensemble Spearman 0.241 EPIT, 0.227 Soccol), yet Soccol uses formula weight 0.663 and no generic tasks.
EPIT uses 0.264 within its informed half, with the other half generic.</li>
<li><strong>A real feature-support failure, but only part of the gap:</strong> hiding the 11 composition columns unsupported
by Soccol improves Soccol from 0.388 to 0.442; EPIT remains at 0.679 under the same restriction.</li>
<li><strong>The remaining failure is substantive:</strong> after controlling for study and evaluation fold in rank space,
EPIT's prediction–target correlation is 0.518 and Soccol's is 0.004. The ordinary pooled correlations are 0.724 and 0.416.</li>
</ul>
<p class="note">Model comparisons use step 5000, the latest checkpoint shared by all trained models, fixed before these diagnostics.
All original baselines were reproduced against the previous run. No specialist or COMB checkpoint was retrained or chosen using these results.
Additional CatBoost probes use fixed settings and the existing outer query partitions.</p></section>

<section><h2>1. What is shared, and what changes?</h2>
<p>All three tables concern reported pitting potential in millivolts and express the retained compositions in wt%.
EPIT and Soccol share substantial Fe–Cr–Ni–Mo chemistry and chloride environments. Their observed median temperature is
25 °C in both; median chloride is 0.5 M in both. Their median reported pH values are 7.0 and 6.5.
They are not wholly unrelated corrosion tasks. Soccol's Fe value is an approximate reconstructed remainder in 3,777/4,027 rows,
so its apparent composition closure is not independent confirmation of a fully measured alloy.</p>
{table(pd.DataFrame(overview),1)}
<p>The analyzed cohorts are the exact model inputs: 760 EPIT rows with numeric targets; 4,027 uncensored/event-1 Soccol rows;
267 source-screened MPEA rows. These are filtered cohorts, not the entire original workbooks. Study counts are source IDs,
not proof of mutually independent publications. MPEA has 116 frozen composition groups.</p>
<p>MPEA's largest element has a median share of only 25.3 wt%, versus about 68–70 wt% in the source tables.
Only one MPEA row has Fe ≥50 wt%. Its target distribution is much wider: IQR 927 mV, versus 477 mV in EPIT and 354 mV in Soccol.
Those shifts are real, but raw range shifts alone do not establish a failure mechanism: input/target standardization removes some scale differences.</p>
{image("distributions.png","Empirical distributions use observed values only. Soccol has no Co feature; no missing Co value is treated as an observed zero.")}
</section>

<section><h2>2. Raw dataset similarity is not the same as training-prior similarity</h2>
<p><strong>Soccol has no Al, Co, Cu, W, Ta, B, Mg, Y, Ce, Gd or Re composition channel in its informed generator.</strong>
At least one of these elements is present in 240/267 MPEA rows (89.9%). Their combined mass has median 27.0 wt% and
interquartile range 14.0–35.6 wt%. MPEA reports Co in 69.7%, Al in 40.1%, and Cu in 25.5% of rows.
These are important parts of the composition representation, not occasional trace omissions.</p>
<p>However, <strong>“EPIT succeeds because its informed generator covers MPEA alloys” is contradicted by the actual sampler.</strong>
The full EPIT dataset is broader, but the selected feature profile restricts informed compositions to Fe and Ni–Cr–Mo families.
The effective probabilities, after selecting these two entries from the five-family configuration and renormalizing, are
94.6% Fe and 5.4% Ni–Cr–Mo. The other families in that configuration are not sampled by this feature profile.
In the previous 131,072-row synthetic audit, EPIT-prior Co never exceeded 0.27 wt%; MPEA's median is 20.2 wt%.</p>
{image("prior_coverage.png","Actual informed sampling: 12,000 rows per source using the frozen perturbation and effective family probabilities. Lines show 5–95% intervals.")}
<p>The median nearest-composition distance from MPEA to the full real EPIT table is 30.1 wt%-L1,
versus 70.8 to Soccol. But to the sampled informed priors it is 74.5 for EPIT and 70.8 for Soccol.
Likewise, the mean absolute difference in pairwise Spearman correlations among nine common composition variables is
0.373 for the EPIT prior and 0.328 for Soccol. On these diagnostics, EPIT's informed prior is not closer.</p>
<p class="note">Composition distances use a projection onto MPEA-observed shared elements, with unsupported/unreported entries set
to zero only for this diagnostic. They are not distances between perfectly known full chemistries.
Both generators independently sample composition and environment, and both use Fe/Ni-oriented source chemistry.
Those shared properties cannot by themselves explain EPIT's advantage.</p>
</section>

<section><h2>3. Soccol loses more of its predictive inputs on MPEA</h2>
{image("coverage.png","Observed-value coverage in the actual shared model tables. A missing value remains unknown, not physical zero.")}
<p>Soccol has pH in 98.4%, temperature in 98.8%, and scan rate in 93.5% of rows, plus surface preparation,
aeration, time, area and ion channels. MPEA supplies none of the pH or scan-rate values and only five temperatures.
Of the 188 reported MPEA chloride values, 160 are exactly 0.6 M. Nitrogen and sulfur are also completely absent from MPEA inputs,
although they enter source rule families. EPIT also suffers missing-input shift, but its native function uses far fewer
non-composition channels and its formula contribution is weaker.</p>
<p>A fixed CatBoost probe measures the predictive information lost when input groups are removed:</p>
{table(pd.DataFrame(cb))}
<p>Removing non-composition inputs reduces Spearman by <strong>0.330 on Soccol, 0.137 on EPIT, and 0.074 on MPEA</strong>.
This is consistent with source-specific tuning rewarding greater environmental specialization on Soccol.
It also shows that MPEA still contains useful composition signal despite its missing metadata.</p>
{image("feature_group_probe.png","Fixed CatBoost settings: 1000 iterations, depth 6, learning rate 0.03, L2=3, seed 42. No hyperparameter or feature-set selection.")}
<p class="note">EPIT and Soccol use their original 152- and 805-row final-test partitions, with 608 and 3,222 development-context rows.
MPEA uses its frozen five outer folds. The source scores are single-partition diagnostics; MPEA values are equal fold means.
The ablation is a fixed-model predictive probe, not a causal or additive variance decomposition.
Non-composition inputs can also encode study/protocol cues; their predictive benefit is not exclusively a physical environment effect.
The slightly higher MPEA composition-plus-chloride score is post-hoc evidence, not a selected new benchmark configuration.</p>
</section>

<section><h2>4. Which target relationships carry over?</h2>
{table(rt)}
{image("relationships.png","Univariate pooled Spearman correlations. NA means an absent or insufficiently variable observed feature.")}
<p><strong>Mo remains positively associated with target in all three datasets.</strong> Cr is positively associated in EPIT/Soccol,
but its MPEA association is weak and uncertain: rho −0.129, study-bootstrap interval [−0.438, 0.213].
After centering ranks within studies, the Cr association is 0.265 in EPIT, 0.266 in Soccol, and 0.012 in MPEA.
This does not justify claiming that chromium has reversed its physical effect.</p>
<p><strong>Cu is a particularly clear missing channel for Soccol:</strong> MPEA Cu–target rho is −0.526,
with study-bootstrap interval [−0.641, −0.396], and within-study-centered rank correlation −0.429.
Soccol's informed generator has no Cu input. EPIT's actual MPEA predictions track this negative association
(prediction–Cu rho −0.707), while Soccol's is much weaker (−0.295).
This is evidence of different adaptation to available context, not proof of a causal Cu effect; alloy composition and study design are correlated.</p>
<p>MPEA also has strong pooled Ti/Nb associations, but Nb's within-study-centered association falls to 0.035.
Similarly, Ni's pooled MPEA association is negative while its within-study-centered association is positive.
These reversals show why pooled element correlations alone are inadequate explanations.</p>
<p class="note">Feature intervals resample source studies, 399 replicates. They are exploratory pointwise intervals with no
multiple-comparison correction. “Within-study-centered rank correlation” is Pearson correlation after subtracting study means
from global ranks, not an ordinary within-study Spearman average.</p>
</section>

<section><h2>5. The formulas are weak on MPEA for both sources; their training influence differs</h2>
<table><thead><tr><th>Training property</th><th>EPIT specialist</th><th>Soccol specialist</th></tr></thead><tbody>
<tr><td>Generic synthetic tasks</td><td>50%</td><td>0%</td></tr>
<tr><td>Formula coefficient within informed target blend</td><td>0.264</td><td>0.663</td></tr>
<tr><td>Informed MLP/tree task mix</td><td>70/30</td><td>100/0</td></tr>
<tr><td>Composition perturbation strength</td><td>0.150</td><td>0.036</td></tr>
<tr><td>Coefficient variation</td><td>0.8</td><td>0.8</td></tr>
<tr><td>Actual formula ensemble / MPEA target rho</td><td>0.241</td><td>0.227</td></tr>
<tr><td>Formula rho with environment fixed at common values</td><td>0.250</td><td>0.201</td></tr>
</tbody></table>
<p>The formula diagnostic calls the <em>actual synthetic-target implementation</em>, with 128 sampled rule/coefficient draws per source.
Missing MPEA inputs receive source-profile means; a second sensitivity fixes temperature/pH/chloride to 25 °C/7/0.6 M
and N/S/selected ions to zero. These are explicit surrogate inputs, not claims about unreported measurements.
No coefficients, family probabilities or imputation values were fitted to MPEA labels.</p>
<p>The key difference is therefore <strong>not uniquely bad Soccol formulas</strong>: both rule mixtures transfer weakly.
It is the much stronger exposure to that limited signal in Soccol's training recipe, together with no generic tasks,
less composition perturbation, and different SCM diversity. In synthetic feature probes, full EPIT rule scores correlate
0.776 with composition-only variation and 0.521 with environment-only variation; the corresponding Soccol values are
0.624 and 0.671. These correlations describe reliance, not additive variance shares.</p>
<p>The observed checkpoint trend agrees with over-specialization: Soccol's MPEA mean rho falls from 0.497 at step 1000
to 0.388 at step 5000, while EPIT rises from 0.666 to 0.727. COMB 50–25–25 reaches 0.692 at step 5000,
above COMB v2's 0.656 but below EPIT. These comparisons change several training ingredients together;
they do not isolate formula weight from generic-task fraction or generator design.</p>
</section>

<section><h2>6. Controlled intervention: remove Soccol-unsupported composition columns</h2>
<p>The same eleven columns were set to missing in both context and query for all three frozen checkpoints.
The 50-column order, weights, labels, and outer query rows were unchanged.</p>
{table(pd.DataFrame(diagnostic))}
{image("composition_ablation.png","Each line connects the same outer fold before and after hiding the eleven composition columns. Bars mark five-fold means.")}
<p>Soccol improves in <strong>all five folds</strong>, demonstrating sensitivity to those out-of-support inputs.
But its RMSE barely changes, and EPIT still leads by 0.237 mean Spearman when both receive the same restricted information.
Thus unsupported inputs are one contributor, not the full explanation. Hiding them also removes useful MPEA signal,
as shown by the EPIT and COMB declines. This is a diagnostic, not a proposed production fix.</p>
</section>

<section><h2>7. Where Soccol fails on MPEA</h2>
{table(subset)}
<p>These are pooled correlations of predictions from the existing outer folds, so they differ slightly from the equal-fold means above.
The advantage persists within NaCl/seawater, at exactly 0.6 M recorded chloride, and after excluding rows with elements absent
from the shared 50-column schema. The last condition excludes Zr/Hf/Sn/La omissions shared by all models.</p>
<p>Across all rows, EPIT's pooled advantage is 0.308, with a study-bootstrap 95% interval [0.173, 0.494].
Its mean absolute error is 126 mV lower, interval [68, 180] mV. EPIT has lower MAE in 42/60 studies.
Removing any single study from the metric leaves a pooled rho advantage between 0.281 and 0.344.
The failure is not attributable to one publication.</p>
{table(ranks)}
<p>The study-and-fold controls compare rows against the same labeled context while removing between-study level differences.
Both rank definitions show that Soccol has very little remaining ordering skill. EPIT retains substantial signal.
This is stronger evidence than a small change in average test score.</p>
{image("prediction_scatter.png","Original predictions from the fixed folds; no predictions averaged across models or fitted using query labels.")}
<p>The 27 rows with no Soccol-unsupported elements have similar pooled EPIT/Soccol ranking (0.666/0.663),
whereas the other 240 rows have 0.700/0.344. However, the small subset spans only seven studies and includes
Ti/Zr/Hf-rich alloys outside Soccol's source distribution. It is not a matched in-domain control, and cannot establish
that the omitted columns alone cause the gap.</p>
</section>

<section><h2>8. Measurement and provenance issues: important, but not unique explanations</h2>
{table(env,1)}
<p>MPEA's upper tail is partly concentrated in nitric-acid observations: 17 rows from two studies, median 1854 mV.
The model table does not retain the raw electrolyte label, although chloride missingness can carry indirect information.
Matching chloride concentration does not match pH, temperature, preparation or measurement protocol.</p>
<p>MPEA also has 147 rows in 34 identical-input groups; some identical model inputs have target differences up to 1456 mV.
Unrecorded process, phase, environment or measurement differences limit all models. They cannot on their own explain
why EPIT beats Soccol on identical inputs. Likewise, 66/267 MPEA rows lose some composition mass because Zr/Hf/Sn/La are
outside the shared schema; excluding those rows does not remove the gap.</p>
<p>EPIT and MPEA targets are explicitly SCE. Soccol has 565 verified SCE rows, 112 verified Ag/AgCl rows,
and 3350 with unresolved source-level reference scale in the current conservative audit.
No electrode conversion was applied. This can affect source calibration and mixed-source transfer.
It cannot be the direct context-scale explanation for the new MPEA-only gap: both specialists receive identical SCE context labels.
A uniform reference offset would also leave within-source Spearman ordering unchanged.</p>
</section>

<section><h2>Conclusion</h2>
<p><strong>The data support a transfer failure of the Soccol training regime, not a general inability to learn corrosion.</strong>
Soccol's source task makes rich environmental information and a tightly constrained source prior useful.
MPEA removes most of that information and introduces major composition channels outside Soccol's informed schema.
EPIT's more generic, less formula-dominated recipe adapts substantially better to the resulting context task,
even though its informed chemistry is also out of distribution.</p>
<p><strong>Directly demonstrated:</strong> missing predictive channels; unsupported-feature sensitivity; weak formula compatibility;
a large, study-robust performance gap; and near-zero Soccol within-study ranking.
<strong>Best-supported inference:</strong> Soccol is over-specialized to its source information and relationships.
<strong>Not yet identified:</strong> the individual causal share of formula weight, generic-task absence, SCM diversity,
perturbation, or their interaction. No claim that one parameter alone is proven responsible is warranted.</p>
</section>

<section><h2>Reproducibility and boundaries</h2>
<p>This is exploratory analysis of already inspected datasets and checkpoints, not a fresh independent confirmation.
The MPEA folds separate composition groups, not studies. Study-bootstrap intervals are conditional on saved predictions;
training is not repeated in bootstrap samples. All measurement references and model outcomes remain unchanged.</p>
<p>Run the scripts in this folder from the repository's virtual environment, in this order:
<code>reproduce.py</code>, <code>model_diagnostic.py</code>, <code>rule_diagnostic.py</code>,
<code>feature_group_diagnostic.py</code>, <code>analyze_predictions.py</code>, then <code>build_report.py</code>.
The first five compute diagnostics; the sixth builds this report.
The CPU model diagnostic uses eight threads by default. Input hashes, settings, source paths and limitations are saved alongside results.</p>
<p>Key data files:
<a href="dataset_summary.csv">dataset summary</a> ·
<a href="feature_distributions.csv">all feature distributions</a> ·
<a href="target_relationships.csv">study-aware target relationships</a> ·
<a href="prior_profiles.json">actual prior profiles</a> ·
<a href="sampled_prior_feature_coverage.csv">sampled prior coverage</a> ·
<a href="rule_scores.csv">formula scores</a> ·
<a href="catboost_feature_group_metrics.csv">feature-group probe</a> ·
<a href="model_diagnostics/fold_metrics.csv">fixed-model ablations</a> ·
<a href="paired_study_bootstrap.csv">paired uncertainty estimates</a> ·
<a href="mpea_predictions_with_metadata.csv">predictions and row metadata</a>.</p>
</section>
"""
    document="""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPIT, Soccol and MPEA: dataset and transfer analysis</title><style>
body{margin:0;background:#f6f7f9;color:#202a34;font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1130px;margin:auto;padding:40px 32px 80px;background:white}
h1{font-size:35px;line-height:1.2;letter-spacing:-.7px}h2{font-size:24px;line-height:1.35;margin-top:0}
header{border-bottom:3px solid #2879b5;padding-bottom:24px}.eyebrow{color:#496377;font-size:13px;text-transform:uppercase;letter-spacing:1.1px}
.lead{font-size:19px}.note{font-size:14px;background:#f1f5f8;border-left:3px solid #8aa7bb;padding:12px 18px}
section{padding:28px 0;border-bottom:1px solid #dce3e8}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;max-width:100%;margin:18px 0}
th,td{padding:9px 11px;border-bottom:1px solid #dce3e8;text-align:left}th{background:#eaf1f6;white-space:normal}
tbody tr:nth-child(even){background:#f8fafb}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#566876}
a{color:#165d91}code{background:#edf1f4;padding:2px 4px}li{margin:8px 0}
@media print{body{background:white}main{padding:0;max-width:none}section{break-inside:auto}figure,table{break-inside:avoid}}
</style></head><body><main>"""+body+"</main></body></html>"
    (OUT/"report.html").write_text(document)
    print("Saved",OUT/"report.html")


if __name__=="__main__":
    main()
