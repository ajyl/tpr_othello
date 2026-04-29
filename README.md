

# Train probes:
```
python src/train_probe.py
python src/train_tpr_probe.py
python src/train_multilinear_tpr_probe.py
```


# Intervention with linear probe:
```
python src/intervene_probe.py --probe-pair 6=probes/linear/resid_6_linear.pth --probe-pair 7=probes/linear/resid_7_linear.pth --probe-pair 5=probes/linear/resid_5_linear.pth --probe-pair 4=probes/linear/resid_4_linear.pth --probe-pair 3=probes/linear/resid_3_linear.pth --probe-pair 2=probes/linear/resid_2_linear.pth --probe-pair 1=probes/linear/resid_1_linear.pth --probe-pair 0=probes/linear/resid_0_linear.pth --num-intervened-squares 2
```

# Intervention with TPR probe:
```
python src/intervene_tpr_probe.py --probe-pair 0=probes/tpr/r52_f2/resid_0_tpr_r52_f2_seed1111.pth --probe-pair 1=probes/tpr/r52_f2/resid_1_tpr_r52_f2_seed1111.pth --probe-pair 2=probes/tpr/r52_f2/resid_2_tpr_r52_f2_seed1111.pth --probe-pair 3=probes/tpr/r52_f2/resid_3_tpr_r52_f2_seed1111.pth --probe-pair 4=probes/tpr/r52_f2/resid_4_tpr_r52_f2_seed1111.pth --probe-pair 5=probes/tpr/r52_f2/resid_5_tpr_r52_f2_seed1111.pth --probe-pair 6=probes/tpr/r52_f2/resid_6_tpr_r52_f2_seed1111.pth --probe-pair 7=probes/tpr/r52_f2/resid_7_tpr_r52_f2_seed1111.pth --num-intervened-squares 1
```

# Intervention with Trilinear TPR probe:
```
python src/intervene_multilinear_tpr_probe.py --probe-pair 0=probes/tpr_multilinear/row8_col8_color2/resid_0_mltpr_row8_col8_color2_seed1111.pth --probe-pair 1=probes/tpr_multilinear/row8_col8_color2/resid_1_mltpr_row8_col8_color2_seed1111.pth --probe-pair 2=probes/tpr_multilinear/row8_col8_color2/resid_2_mltpr_row8_col8_color2_seed1111.pth --probe-pair 3=probes/tpr_multilinear/row8_col8_color2/resid_3_mltpr_row8_col8_color2_seed1111.pth --probe-pair 4=probes/tpr_multilinear/row8_col8_color2/resid_4_mltpr_row8_col8_color2_seed1111.pth --probe-pair 5=probes/tpr_multilinear/row8_col8_color2/resid_5_mltpr_row8_col8_color2_seed1111.pth --probe-pair 6=probes/tpr_multilinear/row8_col8_color2/resid_6_mltpr_row8_col8_color2_seed1111.pth --probe-pair 7=probes/tpr_multilinear/row8_col8_color2/resid_7_mltpr_row8_col8_color2_seed1111.pth
```

# Plot isomap:
```
python plot_scripts/plot_tpr_embedding_isomap.py probes/tpr/r52_f2/resid_6_tpr_r52_f2_seed1111.pth --output-path isomap --exclude-center-squares
```

# Plot binding PCA:
```
python plot_scripts/plot_tpr_embedding_binding.py --probe-path probes/tpr/r52_f2/resid_6_tpr_r52_f2_seed1111.pth --output-path binding.pdf
```

# Plot Gram matrices:
```
python src/gram_matrix.py --probe-path probes/tpr_multilinear/u8_v8_f2/resid_6_mltpr_row8_col8_color2_seed1111.pth --probe-path2 probes/tpr_multilinear/u6_v8_f2/resid_6_mltpr_row6_col8_color2_seed1111.pth --factor row --output-path gram.pdf
```

# Plot TPR vs. Linear probes:
```
python plot_scripts/tpr_vs_linear.py --linear-probe-path-top probes/linear/resid_6_linear.pth --distributed-linear-probe-path probes/linear/resid_6_linear.pth --local-tpr-probe-path probes/tpr/r64_f2/resid_6_tpr_r64_f2_seed1111.pth --distributed-tpr-probe-path probes/tpr/r56_f2/resid_6_tpr_r56_f2_seed1111.pth --heatmap-output-path tpr_vs_linear.pdf
```


# Plot TPR vs. SVD:
```
python plot_scripts/tpr_vs_svd.py --linear-probe-path probes/linear/resid_6_linear.pth --tpr-probe-path probes/tpr/r52_f2/resid_6_tpr_r52_f2_seed1111.pth --plot-output-path tpr_vs_svd.pdf --num-plot-points 40 --x-axis-units percentage
```

# Plot local k-NN based geometry of TPR:
```
python plot_scripts/local_geometry_knn.py --tpr-probe-path probes/tpr/r52_f2/resid_6_tpr_r52_f2_seed1111.pth --baseline-probe-path probes/tpr_baseline/tpr_baseline_d512_r52_f2_reprseed1111_seed1111.pth --iid-baseline-probe-path probes/tpr_baseline/tpr_baseline_d512_r52_f2_reprseed1111_iidboards_seed1111.pth --match-groundtruth-degree --include-diagonals --metric cosine --as-percentages --output-path local_geometry_knn.pdf
```

# Plot pairwise gap distance heatmaps:
```
python plot_scripts/local_geometry_heatmap.py --probe-path probes/tpr/r52_f2/resid_6_tpr_r52_f2_seed1111.pth --baseline-probe-path probes/tpr_baseline/tpr_baseline_d512_r52_f2_reprseed1111_seed1111.pth --iid-baseline-probe-path probes/tpr_baseline/tpr_baseline_d512_r52_f2_reprseed1111_iidboards_seed1111.pth --metric cosine --include-diagonals --gap-grid-plot-path grid.pdf
```
