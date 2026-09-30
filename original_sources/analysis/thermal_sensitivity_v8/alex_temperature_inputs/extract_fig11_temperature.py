"""Read published Fig. 11 vector geometry; never treat values as raw telemetry.

Run with the Codex bundled Python runtime. Outputs are analysis intermediates.
"""
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
import csv
import json
import math

import pdfplumber


SOURCE = Path(r"D:\月球导航\data\external_reference\lugre_antenna\LuGRE_first_results_NAVIGATION_2026_official.pdf")
OUT = Path(__file__).resolve().parent
OPS = [1, 2, 3, 5, 9, 12, 14, 17, 18, 21, 22, 23, 27, 31, 32, 37]
COLORS = {
    (0.467, 0.671, 0.259): "receiver",
    (0.780, 0.090, 0.502): "HGA",
    (0.298, 0.580, 0.816): "LNA",
}


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with pdfplumber.open(SOURCE) as doc:
        page = doc.pages[16]
        # The top and bottom axis rules are visually verified against the
        # Fig. 11 tick labels 60 and -20 degrees Celsius.
        axis_rules = [l for l in page.lines
                      if 350 < l["width"] < 360 and l["height"] < 0.01
                      and tuple(l["stroking_color"]) == (0.149, 0.149, 0.149)
                      and 290 < l["top"] < 435]
        assert len(axis_rules) == 2, axis_rules
        top, bottom = sorted(l["top"] for l in axis_rules)

        def temp_c(y):
            return 60 - (y - top) * 80 / (bottom - top)

        spans = defaultdict(list)
        means = defaultdict(list)
        markers = defaultdict(list)
        for item in page.lines:
            color = tuple(item["stroking_color"] or ())
            if (color in COLORS and item["width"] < 0.001
                    and 140 < item["x0"] < 468
                    and 290 < item["top"] < item["bottom"] < 435):
                spans[COLORS[color]].append(item)
        for item in page.curves:
            color = tuple(item["non_stroking_color"] or ())
            if not (color in COLORS and 140 < item["x0"] < 468
                    and 290 < item["top"] < item["bottom"] < 435):
                continue
            if len(item["pts"]) == 5 and 5.3 < item["width"] < 5.6:
                # Legend markers are outside the operation x grid; excluded
                # below by exact matching to colored vertical range lines.
                means[COLORS[color]].append(item)
            if len(item["pts"]) == 3:
                markers[COLORS[color]].append(item)

        rows = []
        audit = []
        triangle_errors = []
        for component in ["HGA", "LNA", "receiver"]:
            ranges = sorted(spans[component], key=lambda x: x["x0"])
            assert len(ranges) == 16, (component, len(ranges))
            for op, line in zip(OPS, ranges):
                candidates = [m for m in means[component]
                              if abs((m["x0"] + m["x1"]) / 2 - line["x0"]) < 0.01]
                assert len(candidates) == 1, (component, op, len(candidates))
                m = candidates[0]
                ymean = (m["top"] + m["bottom"]) / 2
                group_markers = [m for m in markers[component]
                                 if abs(sum(q[0] for q in m["pts"]) / 3 - line["x0"]) < 0.01]
                assert len(group_markers) == 2, (component, op, len(group_markers))
                tri_ys = sorted(sum(q[1] for q in m["pts"]) / 3 for m in group_markers)
                triangle_errors.extend([abs(tri_ys[0] - line["top"]),
                                        abs(tri_ys[1] - line["bottom"])])
                low, mean, high = temp_c(line["bottom"]), temp_c(ymean), temp_c(line["top"])
                assert low <= mean <= high, (component, op, low, mean, high)
                row = {"operation": f"OP{op}_0", "operation_model_label": f"OP{op}",
                       "component": component,
                       "minimum_c": round(low, 6), "mean_c": round(mean, 6), "maximum_c": round(high, 6),
                       "minimum_k": round(low + 273.15, 6), "mean_k": round(mean + 273.15, 6),
                       "maximum_k": round(high + 273.15, 6),
                       "source_kind": "publication_vector_figure_digitization",
                       "source_pdf_page": 17, "source_figure": "11",
                       "time_resolution": "per_operation_summary_not_time_series",
                       "suggested_digitization_sensitivity_c": 0.5}
                rows.append(row)
                audit.append({"operation": row["operation"], "component": component,
                              "x_pdf_pt": line["x0"], "min_y_pdf_pt": line["bottom"],
                              "mean_y_pdf_pt": ymean, "max_y_pdf_pt": line["top"],
                              "min_triangle_centroid_y_pdf_pt": tri_ys[1],
                              "max_triangle_centroid_y_pdf_pt": tri_ys[0]})

    rows.sort(key=lambda r: (OPS.index(int(r["operation_model_label"][2:])), r["component"]))
    write_csv("fig11_operation_temperature_long.csv", rows)
    wide = []
    for op in OPS:
        current = {"operation": f"OP{op}_0", "operation_model_label": f"OP{op}"}
        for row in [r for r in rows if r["operation"] == current["operation"]]:
            for stat in ["minimum", "mean", "maximum"]:
                current[f"{row['component']}_{stat}_c"] = round(row[f"{stat}_c"], 1)
        wide.append(current)
    write_csv("fig11_operation_temperature_display.csv", wide)

    by_op = {(r["operation"], r["component"]): r for r in rows}
    sensitivities = []
    reference = {}
    for band, nf_db, nominal_k in [("L1_E1", 0.8, 182.0), ("L5_E5a", 1.3, 231.0)]:
        excess_factor = 10 ** (nf_db / 10) - 1
        # The high-gain approximation reduces to THGA + (F - 1) * TLNA.
        ref = sum(by_op[(f"OP{op}_0", "HGA")]["mean_k"] +
                  excess_factor * by_op[(f"OP{op}_0", "LNA")]["mean_k"]
                  for op in OPS) / len(OPS)
        reference[band] = ref
        for op in OPS:
            opid = f"OP{op}_0"
            for stat in ["minimum", "mean", "maximum"]:
                hga = by_op[(opid, "HGA")][f"{stat}_k"]
                lna = by_op[(opid, "LNA")][f"{stat}_k"]
                tsys = hga + excess_factor * lna
                delta = -10 * math.log10(tsys / ref)
                sensitivities.append({
                    "operation": opid, "band": band, "scenario": stat,
                    "hga_temperature_k": hga, "lna_temperature_k": lna,
                    "lna_noise_figure_db": nf_db,
                    "alex_simplified_tsys_proxy_k": tsys,
                    "alex_reference_mean_across_16_operations_k": ref,
                    "delta_cn0_alex_relative_reference_db": delta,
                    "fixed_baseline_reference_k": nominal_k,
                    "fixed_baseline_times_relative_thermal_ratio_k": nominal_k * tsys / ref,
                    "absolute_proxy_replacement_delta_db_not_recommended": -10 * math.log10(tsys / nominal_k),
                    "interpretation": "relative_sensitivity_scenario_not_calibrated_noise_temperature",
                })
    write_csv("alex_relative_thermal_sensitivity.csv", sensitivities)
    metadata = {
        "source_pdf": str(SOURCE), "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "pdf_page": 17, "figure": 11, "operations_in_plot": [f"OP{n}_0" for n in OPS],
        "plot_axis_pdf_top_pt": top, "plot_axis_pdf_bottom_pt": bottom,
        "plot_axis_top_temperature_c": 60, "plot_axis_bottom_temperature_c": -20,
        "method": "PDF vector range-line endpoints and circular marker centres; triangle centroids are independent checks",
        "max_range_endpoint_vs_triangle_centroid_disagreement_pdf_pt": max(triangle_errors),
        "max_range_endpoint_vs_triangle_centroid_disagreement_c": max(triangle_errors) * 80 / (bottom - top),
        "digitization_sensitivity_c": 0.5,
        "digitization_sensitivity_status": "analyst_selected_rounding/graphical_sensitivity_not_statistical_CI_or_sensor_accuracy",
        "precision_note": "Six decimals preserve coordinate transformation reproducibility, not telemetry precision; display to 0.1 C only.",
        "alex_reference_system_temperature_proxy_k": reference,
        "model": "Tsys_proxy = THGA_K + (10**(NF_dB/10)-1)*TLNA_K",
        "delta_reference": "unweighted mean of 16 per-operation mean-scenario proxy values, separately per band",
        "surface_note": "Published Fig. 12 contains surface temperature curves, extracted separately into surface_series.csv.",
        "limits": ["No raw numerical temperature telemetry is recovered.",
                   "Fig. 11 is per-operation min/mean/max, not a synchronized time series.",
                   "Sensor placement does not guarantee actual component temperature.",
                   "HGA physical temperature is not calibrated antenna noise temperature.",
                   "Receiver temperatures are extracted but the receiver after LNA is omitted in the high-gain approximation.",
                   "Do not substitute this absolute Tsys proxy directly for 182/231 K without an antenna/receiver noise model.",
                   "The min/mean/max paired scenarios are not confidence intervals or proven simultaneous extrema."],
    }
    (OUT / "digitization_provenance.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "vector_coordinate_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sensitivity_rows": len(sensitivities), "references_k": reference,
                      "max_geometry_check_c": metadata["max_range_endpoint_vs_triangle_centroid_disagreement_c"],
                      "output_dir": str(OUT)}, ensure_ascii=False))
    print("operation,HGA_min_mean_max_C,LNA_min_mean_max_C,receiver_min_mean_max_C")
    for r in wide:
        print(r["operation"], *["/".join(str(r[f"{c}_{s}_c"]) for s in ["minimum", "mean", "maximum"])
                               for c in ["HGA", "LNA", "receiver"]], sep=",")


if __name__ == "__main__":
    main()
