#!/usr/bin/env python3

import argparse
import json
import re
import statistics
import subprocess
import sys


DEFAULT_SUITES = [
    ("metal_endtoend", r"^testZfpCuda(1d|2d|3d)(Float|Double|Int32|Int64)$"),
    ("core_throughput", r"^testzfp$"),
]

# Data sizes for each end-to-end test.
# Array dimensions are computed by genSmoothRandNums.c starting from sideLen=5,
# doubling-minus-one until total >= MIN_TOTAL_ELEMENTS.
# Float/Double: MIN_TOTAL_ELEMENTS = 1000000
#   1D: 1048577 elems, 2D: 1025x1025=1050625, 3D: 129x129x129=2146689
# Int32/Int64: MIN_TOTAL_ELEMENTS = 4096
#   1D: 4097 elems, 2D: 65x65=4225, 3D: 17x17x17=4913
_TEST_DATA_BYTES = {
    "testZfpCuda1dFloat":  1048577 * 4,
    "testZfpCuda2dFloat":  1050625 * 4,
    "testZfpCuda3dFloat":  2146689 * 4,
    "testZfpCuda4dFloat":  1185921 * 4,
    "testZfpCuda1dDouble": 1048577 * 8,
    "testZfpCuda2dDouble": 1050625 * 8,
    "testZfpCuda3dDouble": 2146689 * 8,
    "testZfpCuda4dDouble": 1185921 * 8,
    "testZfpCuda1dInt32":  4097 * 4,
    "testZfpCuda2dInt32":  4225 * 4,
    "testZfpCuda3dInt32":  4913 * 4,
    "testZfpCuda4dInt32":  6561 * 4,
    "testZfpCuda1dInt64":  4097 * 8,
    "testZfpCuda2dInt64":  4225 * 8,
    "testZfpCuda3dInt64":  4913 * 8,
    "testZfpCuda4dInt64":  6561 * 8,
}


def _seconds_to_gb_s(seconds, data_bytes):
    """Convert a list of timing samples (seconds) to GB/s throughput."""
    if not seconds or not data_bytes:
        return []
    gb = data_bytes / (1024.0 ** 3)
    return [gb / s for s in seconds if s > 0]


def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize(values):
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "max": max(values),
    }


def parse_metrics(output):
    start_re = re.compile(r"Start\s+\d+:\s+(\S+)")
    compress_re = re.compile(r"Compress time \(s\):\s*([0-9]*\.?[0-9]+)")
    decompress_re = re.compile(r"Decompress time \(s\):\s*([0-9]*\.?[0-9]+)")
    throughput_re = re.compile(r"throughput=\s*([0-9]*\.?[0-9]+)\s*MB/s")
    testing_re = re.compile(r"^testing\s+(\dD)\s+array\s+of\s+(floats|doubles)", re.IGNORECASE)

    current_test = None
    current_group = ""
    metrics = {}

    for line in output.splitlines():
        m = start_re.search(line)
        if m:
            current_test = m.group(1)
            metrics.setdefault(current_test, {"compress_s": [], "decompress_s": [], "throughput_mb_s": []})
            continue

        m = testing_re.search(line.strip())
        if m:
            current_group = f"testzfp_{m.group(1).lower()}_{m.group(2).lower()}"
            metrics.setdefault(current_group, {"compress_s": [], "decompress_s": [], "throughput_mb_s": []})
            continue

        if current_test is None and not current_group:
            continue

        metric_target = current_test if current_test is not None else current_group

        m = compress_re.search(line)
        if m:
            metrics[metric_target]["compress_s"].append(float(m.group(1)))
            continue

        m = decompress_re.search(line)
        if m:
            metrics[metric_target]["decompress_s"].append(float(m.group(1)))
            continue

        m = throughput_re.search(line)
        if m:
            metrics[metric_target]["throughput_mb_s"].append(float(m.group(1)))

    return metrics


def merge_metrics(dst, src):
    for test_name, series in src.items():
        entry = dst.setdefault(test_name, {"compress_s": [], "decompress_s": [], "throughput_mb_s": []})
        entry["compress_s"].extend(series["compress_s"])
        entry["decompress_s"].extend(series["decompress_s"])
        entry["throughput_mb_s"].extend(series["throughput_mb_s"])


def run_suite(build_dir, suite_name, regex, repeats):
    aggregate = {}
    for i in range(repeats):
        print(f"[run {i + 1}/{repeats}] suite={suite_name} regex={regex}")
        cmd = ["ctest", "--test-dir", build_dir, "-R", regex, "-V"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"ctest failed for suite '{suite_name}' on iteration {i + 1}")
        merge_metrics(aggregate, parse_metrics(proc.stdout))
    return aggregate


def print_summary(all_metrics):
    print("\n=== Performance Summary ===")
    for suite_name in sorted(all_metrics.keys()):
        print(f"\n[{suite_name}]")
        suite_metrics = all_metrics[suite_name]
        if not suite_metrics:
            print("  (no metrics parsed)")
            continue
        for test_name in sorted(suite_metrics.keys()):
            series = suite_metrics[test_name]
            comp = summarize(series["compress_s"])
            decomp = summarize(series["decompress_s"])
            thr = summarize(series["throughput_mb_s"])
            data_bytes = _TEST_DATA_BYTES.get(test_name, 0)

            print(f"- {test_name}")
            if comp:
                if data_bytes:
                    comp_gb_s = _seconds_to_gb_s(series["compress_s"], data_bytes)
                    comp_gb_sum = summarize(comp_gb_s)
                    # peak = max GB/s (from fastest run), typical = median GB/s
                    print(
                        f"  compress:   {comp_gb_sum['max']:.3f} GB/s peak, "
                        f"{comp_gb_sum['median']:.3f} GB/s median  "
                        f"(n={comp['n']}, min_s={comp['min']:.6f})"
                    )
                else:
                    print(
                        "  compress_s: "
                        f"n={comp['n']} min={comp['min']:.6f} med={comp['median']:.6f} "
                        f"p90={comp['p90']:.6f} max={comp['max']:.6f}"
                    )
            if decomp:
                if data_bytes:
                    decomp_gb_s = _seconds_to_gb_s(series["decompress_s"], data_bytes)
                    decomp_gb_sum = summarize(decomp_gb_s)
                    print(
                        f"  decompress: {decomp_gb_sum['max']:.3f} GB/s peak, "
                        f"{decomp_gb_sum['median']:.3f} GB/s median  "
                        f"(n={decomp['n']}, min_s={decomp['min']:.6f})"
                    )
                else:
                    print(
                        "  decompress_s: "
                        f"n={decomp['n']} min={decomp['min']:.6f} med={decomp['median']:.6f} "
                        f"p90={decomp['p90']:.6f} max={decomp['max']:.6f}"
                    )
            if thr:
                print(
                    "  throughput_mb_s: "
                    f"n={thr['n']} min={thr['min']:.2f} med={thr['median']:.2f} "
                    f"p90={thr['p90']:.2f} max={thr['max']:.2f}"
                )


def main():
    parser = argparse.ArgumentParser(description="Run repeated CTest perf suites and print median/p90 summaries")
    parser.add_argument("--build-dir", required=True, help="CTest build directory (e.g. build-metal)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeated CTest runs per suite")
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Custom suite in form name=regex. May be provided multiple times.",
    )
    parser.add_argument("--json-out", default="", help="Optional path to write JSON summary")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    suites = []
    if args.suite:
        for item in args.suite:
            if "=" not in item:
                raise ValueError(f"invalid --suite '{item}', expected name=regex")
            name, regex = item.split("=", 1)
            suites.append((name.strip(), regex.strip()))
    else:
        suites = list(DEFAULT_SUITES)

    all_metrics = {}
    for suite_name, regex in suites:
        all_metrics[suite_name] = run_suite(args.build_dir, suite_name, regex, args.repeats)

    print_summary(all_metrics)

    if args.json_out:
        json_data = {}
        for suite_name, suite_metrics in all_metrics.items():
            json_suite = {}
            for test_name, series in suite_metrics.items():
                entry = dict(series)  # copy raw arrays
                data_bytes = _TEST_DATA_BYTES.get(test_name, 0)
                if data_bytes:
                    entry["data_bytes"] = data_bytes
                    comp_gb_s = _seconds_to_gb_s(series["compress_s"], data_bytes)
                    decomp_gb_s = _seconds_to_gb_s(series["decompress_s"], data_bytes)
                    if comp_gb_s:
                        entry["compress_gb_s"] = comp_gb_s
                        entry["compress_gb_s_summary"] = summarize(comp_gb_s)
                    if decomp_gb_s:
                        entry["decompress_gb_s"] = decomp_gb_s
                        entry["decompress_gb_s_summary"] = summarize(decomp_gb_s)
                comp_sum = summarize(series["compress_s"])
                decomp_sum = summarize(series["decompress_s"])
                if comp_sum:
                    entry["compress_s_summary"] = comp_sum
                if decomp_sum:
                    entry["decompress_s_summary"] = decomp_sum
                json_suite[test_name] = entry
            json_data[suite_name] = json_suite
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
