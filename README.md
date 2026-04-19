

# Train probes:
```
python src/train_probe.py
python src/train_tpr_probe.py

```


# Intervention with linear probe:
```
python src/intervene_probe.py --probe-pair 6=probes/linear/resid_6_linear.pth --probe-pair 7=probes/linear/resid_7_linear.pth --probe-pair 5=probes/linear/resid_5_linear.pth --probe-pair 4=probes/linear/resid_4_linear.pth --probe-pair 3=probes/linear/resid_3_linear.pth --probe-pair 2=probes/linear/resid_2_linear.pth --probe-pair 1=probes/linear/resid_1_linear.pth --probe-pair 0=probes/linear/resid_0_linear.pth --num-intervened-squares 2
```

# Interventions with TPR probe:
```
python src/intervene_tpr_probe.py --probe-pair 0=probes/tpr/r52_f2/resid_0_tpr_r52_f2_seed1111.pth --probe-pair 1=probes/tpr/r52_f2/resid_1_tpr_r52_f2_seed1111.pth --probe-pair 2=probes/tpr/r52_f2/resid_2_tpr_r52_f2_seed1111.pth --probe-pair 3=probes/tpr/r52_f2/resid_3_tpr_r52_f2_seed1111.pth --probe-pair 4=probes/tpr/r52_f2/resid_4_tpr_r52_f2_seed1111.pth --probe-pair 5=probes/tpr/r52_f2/resid_5_tpr_r52_f2_seed1111.pth --probe-pair 6=probes/tpr/r52_f2/resid_6_tpr_r52_f2_seed1111.pth --probe-pair 7=probes/tpr/r52_f2/resid_7_tpr_r52_f2_seed1111.pth --num-intervened-squares 1
```
