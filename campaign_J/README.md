# Campaign J on controlled query corruption

Run `python campaign_j.py` with NumPy, SciPy and Matplotlib to regenerate the two CSV files and the PDF figure. Pass an optional directory to write elsewhere. The experiment was run with NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8. It limits the BLAS runtime to one thread.

`campaign_j_seed_counts.csv` contains one row per noise amplitude and independently generated memory matrix. Each row counts 128 cues. The `recall_*` fields count correct nearest-pattern decodes, with any unresolved run counted as a failure. `recall_raw_cue` gives the zero-update decoding baseline. `resolved_*_flow` counts trials where both the map and final flow pass the stopping criterion. `disagree_*_flow` counts those resolved pairs whose endpoints are farther apart than the matching tolerance. The `flow_audit_endpoint_changes` field compares only pairs resolved under both the early and intermediate flow settings. The `flow_audit_label_changes` field counts changes of the decoded label between those same horizons for all cues, including the ones that had yet to resolve. `campaign_j_summary.csv` pools the counts over the sixteen independent memory matrices in each setting and gives percentile intervals resampled at the memory-matrix level.

`fig_recall_corruption.pdf` is Figure 2 of the article.

## Terminal-state audit

From this directory, run the replay with a separate output directory.

~~~bash
python3 audit_terminal_states.py audit_replayed
~~~

The script evaluates the Hessian at every computed endpoint and reports the
minimum eigenvalue, fixed-point residual, and separation between the pairwise
distance classes. It also calculates the paired confidence interval of the
one-step versus flow recall contrast. The JSON reports aggregate statistics
and the CSV gives the per-bank, per-noise results. The numerical checks refer
to the dimension-32 experiment.
