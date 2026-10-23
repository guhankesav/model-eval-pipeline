"""Before/after numbers for the v2 generator, which applied the v1 domain-gap findings."""
import json
from pathlib import Path

v1 = json.loads(Path("runs/synth-vs-real/report.json").read_text())
v2 = json.loads(Path("runs/synth-v2-vs-real/report.json").read_text())
g1 = json.loads(Path("docs/domain_gap.json").read_text())
g2 = json.loads(Path("docs/domain_gap_v2.json").read_text())

out = {
    "synthetic_trained_accuracy_on_real": {
        m: {"v1": round(v1["domain_gap_by_model"][m]["trained_on_synthetic"], 4),
            "v2": round(v2["domain_gap_by_model"][m]["trained_on_synthetic"], 4)} for m in v1["domain_gap_by_model"]},
    "label_js_bits": {"v1": g1["labels"]["js_divergence_bits"], "v2": g2["labels"]["js_divergence_bits"]},
    "c2st_auc": {"v1": g1["features"]["c2st_auc"], "v2": g2["features"]["c2st_auc"]},
    "frechet": {"v1": g1["features"]["frechet_distance"], "v2": g2["features"]["frechet_distance"],
                "real_vs_real": g1["features"]["frechet_real_vs_real"]},
    "visual_std_mean_diff": {k: {"v1": round(g1["visual"][k]["std_mean_diff"], 2), "v2": round(g2["visual"][k]["std_mean_diff"], 2)}
                             for k in g1["visual"]},
}
Path("docs/generator_v2.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
