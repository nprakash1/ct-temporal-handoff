#!/usr/bin/env python3
"""Create a training-ready, synthetic-text derivative of v6 labels.

The source JSONL is never modified. Every original finding field is preserved. For
known 3-way progression labels, this script adds a deterministic synthetic training
sentence whose semantics match the existing v6 label. Unknown/absent findings receive
no sentence and are marked ineligible for progression losses.

This is deliberately a *weak/silver-label* transformation: synthetic text fixes the
text/label mismatch, but does not independently validate whether the v6 label is true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SEED = 2026
KNOWN_DIRECTIONS = {"worsened", "stable", "improved"}
CHANGE_TO_DIRECTION = {
    "new": "worsened",
    "worse": "worsened",
    "stable": "stable",
    "improved": "improved",
    "resolved": "improved",
}

TEMPLATES = {
    "new": [
        "The {finding} is newly present compared with the prior examination.",
        "A new {finding} is present on the current examination.",
        "Compared with the prior examination, the {finding} is newly present.",
    ],
    "worse": [
        "The {finding} has worsened compared with the prior examination.",
        "There has been interval worsening of the {finding}.",
        "The {finding} has progressed since the previous examination.",
    ],
    "stable": [
        "The {finding} is unchanged compared with the prior examination.",
        "There has been no significant interval change in the {finding}.",
        "The {finding} remains stable compared with the previous examination.",
        "The {finding} is stable compared with the prior study.",
    ],
    "improved": [
        "The {finding} has improved compared with the prior examination.",
        "There has been interval improvement in the {finding}.",
        "The {finding} has decreased since the previous examination.",
    ],
    "resolved": [
        "The {finding} has resolved compared with the prior examination.",
        "The previously present {finding} is no longer seen.",
        "Compared with the prior examination, the {finding} is no longer present.",
    ],
}


def choose_template(record: dict, finding: dict) -> tuple[str, str, int]:
    direction = finding["direction"]
    original_change = finding.get("change")
    # Preserve clinically useful 5-class nuance when it agrees with direction.
    change = (
        original_change
        if CHANGE_TO_DIRECTION.get(original_change) == direction
        else {"worsened": "worse", "stable": "stable", "improved": "improved"}[direction]
    )
    bank = TEMPLATES[change]
    key = "|".join(
        [
            str(SEED),
            str(record.get("patient", "")),
            str(record.get("prior_volume", "")),
            str(record.get("curr_volume", "")),
            str(finding.get("finding", "")),
            direction,
            change,
        ]
    )
    template_id = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(bank)
    sentence = bank[template_id].format(finding=finding["finding"].lower())
    return sentence, change, template_id


def convert_record(record: dict) -> dict:
    # Round-trip copy guarantees the input object is not modified by aliasing.
    output = json.loads(json.dumps(record))
    output["derived_label_version"] = "v6_weak_synthetic_direction_v1"
    output["synthetic_direction_seed"] = SEED
    record_quality_eligible = bool(record.get("parse_ok")) and bool(record.get("schema_ok"))
    output["record_quality_eligible"] = record_quality_eligible

    for finding in output.get("findings", []):
        direction = finding.get("direction")
        eligible = record_quality_eligible and direction in KNOWN_DIRECTIONS
        finding["progression_eligible"] = eligible
        finding["original_label_source"] = finding.get("label_source")
        finding["original_temporal_sentence_source"] = finding.get("temporal_sentence_source")

        if eligible:
            sentence, template_group, template_id = choose_template(output, finding)
            finding["training_temporal_sentence"] = sentence
            finding["training_temporal_sentence_source"] = "synthetic_direction"
            finding["training_template_group"] = template_group
            finding["training_template_id"] = template_id
        else:
            finding["training_temporal_sentence"] = ""
            finding["training_temporal_sentence_source"] = "none"
            finding["training_template_group"] = None
            finding["training_template_id"] = None
    return output


def convert(input_path: Path, output_path: Path) -> dict:
    stats = {
        "records": 0,
        "parse_ok": 0,
        "schema_ok": 0,
        "direction": Counter(),
        "eligible_by_source": Counter(),
        "eligible_by_domain": defaultdict(Counter),
        "template_group": Counter(),
        "template_id": Counter(),
        "findings_per_record": Counter(),
        "record_quality": Counter(),
    }
    seen_pairs = set()

    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record.get("patient"), record.get("prior_volume"), record.get("curr_volume"))
            if key in seen_pairs:
                raise ValueError(f"Duplicate pair at line {line_number}: {key}")
            seen_pairs.add(key)

            output = convert_record(record)
            # Determinism invariant: same source record must produce identical derived fields.
            assert output == convert_record(record), f"Non-deterministic conversion at line {line_number}"
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")

            stats["records"] += 1
            stats["parse_ok"] += bool(record.get("parse_ok"))
            stats["schema_ok"] += bool(record.get("schema_ok"))
            stats["findings_per_record"][len(output.get("findings", []))] += 1
            stats["record_quality"]["eligible" if output["record_quality_eligible"] else "excluded"] += 1
            domain = "train" if str(record.get("prior_volume", "")).startswith("train_") else (
                "valid" if str(record.get("prior_volume", "")).startswith("valid_") else "other"
            )
            for finding in output.get("findings", []):
                direction = finding.get("direction", "missing")
                stats["direction"][direction] += 1
                if finding["progression_eligible"]:
                    stats["eligible_by_source"][finding.get("original_label_source", "missing")] += 1
                    stats["eligible_by_domain"][domain][direction] += 1
                    stats["template_group"][finding["training_template_group"]] += 1
                    stats["template_id"][(finding["training_template_group"], finding["training_template_id"])] += 1

    return stats


def validate_output(input_path: Path, output_path: Path) -> None:
    input_records = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    output_records = [json.loads(line) for line in output_path.open(encoding="utf-8") if line.strip()]
    assert len(input_records) == len(output_records)

    for original, derived in zip(input_records, output_records):
        assert len(original.get("findings", [])) == len(derived.get("findings", []))
        expected_record_quality = bool(original.get("parse_ok")) and bool(original.get("schema_ok"))
        assert derived["record_quality_eligible"] == expected_record_quality
        for before, after in zip(original.get("findings", []), derived.get("findings", [])):
            # Existing source fields must remain exactly unchanged.
            for key, value in before.items():
                assert after.get(key) == value, (key, value, after.get(key))
            if expected_record_quality and after["direction"] in KNOWN_DIRECTIONS:
                assert after["progression_eligible"]
                assert after["training_temporal_sentence"]
                assert after["training_temporal_sentence_source"] == "synthetic_direction"
                assert CHANGE_TO_DIRECTION[after["training_template_group"]] == after["direction"]
            else:
                assert not after["progression_eligible"]
                assert after["training_temporal_sentence"] == ""
                assert after["training_temporal_sentence_source"] == "none"


def print_stats(stats: dict, input_path: Path, output_path: Path) -> None:
    print(f"input : {input_path}")
    print(f"output: {output_path}")
    print(f"records={stats['records']:,} parse_ok={stats['parse_ok']:,} schema_ok={stats['schema_ok']:,}")
    print("findings/record:", dict(stats["findings_per_record"]))
    print("record quality:", dict(stats["record_quality"]))
    print("direction:", dict(stats["direction"]))
    known = sum(stats["eligible_by_source"].values())
    print(f"progression eligible={known:,}")
    print("eligible by original source:", dict(stats["eligible_by_source"]))
    print("eligible by domain:")
    for domain, counts in stats["eligible_by_domain"].items():
        print(" ", domain, dict(counts), "total=", sum(counts.values()))
    print("template groups:", dict(stats["template_group"]))
    print("template IDs:", {f"{g}:{i}": n for (g, i), n in sorted(stats["template_id"].items())})
    print("VALIDATION PASSED: originals preserved; known labels have matching synthetic text; unknowns excluded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="medgemma_labels_v6 (1).jsonl")
    parser.add_argument("--output", default="medgemma_labels_v6_synthetic_direction.jsonl")
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("Refusing to overwrite the input JSONL")
    stats = convert(input_path, output_path)
    validate_output(input_path, output_path)
    print_stats(stats, input_path, output_path)


if __name__ == "__main__":
    main()