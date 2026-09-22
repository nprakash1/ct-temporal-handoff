#!/usr/bin/env python3
import json

P = "notebooks/label_per_finding_temporal_colab.ipynb"
nb = json.load(open(P, encoding="utf-8"))
assert nb["nbformat"] == 4 and len(nb["cells"]) == 11

errors = []
for i, cell in enumerate(nb["cells"]):
    assert "source" in cell
    if cell["cell_type"] != "code":
        continue
    text = "".join(cell["source"])
    if any(line.lstrip().startswith(("!", "%")) for line in text.splitlines()):
        continue
    try:
        compile(text, f"<cell{i}>", "exec")
    except SyntaxError as exc:
        errors.append((i, str(exc)))
assert not errors, errors

all_text = "\n".join("".join(c["source"]) for c in nb["cells"])
prompt_text = "".join(nb["cells"][5]["source"])
assert "TEMPORAL_CUE" not in all_text and "UP_CUE" not in all_text and "DOWN_CUE" not in all_text
assert "import re" not in prompt_text
assert "truncation=False" in all_text and "truncation=True" not in all_text
assert "add_special_tokens=True" in all_text
assert "medgemma_labels_v7.jsonl" in all_text and "LIMIT=50" in all_text
assert "structured_transitions(row)" in all_text  # postprocessing fallback still exists
assert "batch_reached_max_new_tokens" in all_text
assert "cap_parse_fail==0" in all_text
assert "rec['raw']=raw" in all_text and "raw[:" not in "".join(nb["cells"][7]["source"])

namespace = {}
exec(prompt_text, namespace)
canon = namespace["CANON"]
combine = namespace["combine"]

# The LLM prompt must include current text but hide prior text and structured transitions.
probe = {
    "prior_findings": "SECRET PRIOR REPORT CONTENT",
    "prior_impression": "SECRET PRIOR IMPRESSION",
    "curr_findings": "VISIBLE CURRENT REPORT CONTENT",
    "curr_impression": "VISIBLE CURRENT IMPRESSION",
    "presence_changes": json.dumps({"Pleural effusion": "new"}),
    "delta_days": "30",
}
rendered_prompt = namespace["build_user"](probe)
assert "VISIBLE CURRENT REPORT CONTENT" in rendered_prompt
assert "SECRET PRIOR REPORT CONTENT" not in rendered_prompt
assert "SECRET PRIOR IMPRESSION" not in rendered_prompt
assert '"Pleural effusion": "new"' not in rendered_prompt
assert "STRUCTURED TRANSITIONS" not in rendered_prompt


def make_row(presence, current=""):
    return {
        "prior_volume": "train_1_a_1.nii.gz",
        "curr_volume": "train_1_b_1.nii.gz",
        "prior_findings": "",
        "prior_impression": "",
        "curr_findings": current,
        "curr_impression": "",
        "presence_changes": json.dumps(presence),
        "delta_days": "30",
    }


def make_parsed(overrides):
    findings = []
    for finding in canon:
        item = {
            "finding": finding,
            "report_change": "not_temporal",
            "change": "absent",
            "direction": "unknown",
            "label_source": "absent",
            "evidence": "",
            "temporal_sentence": "",
        }
        item.update(overrides.get(finding, {}))
        findings.append(item)
    return {"findings": findings}


def finding(output, name):
    return next(item for item in output if item["finding"] == name)


# Explicit worse, verbatim report sentence.
sentence = "Interval increase in the right pleural effusion."
row = make_row({"Pleural effusion": "present_both"}, sentence)
output, schema_ok, duplicates = combine(row, make_parsed({"Pleural effusion": {
    "report_change": "worse", "evidence": "Interval increase", "temporal_sentence": sentence}}))
item = finding(output, "Pleural effusion")
assert schema_ok and not duplicates
assert (item["direction"], item["label_source"], item["temporal_sentence_source"]) == (
    "worsened", "report_explicit", "real_report")

# Unrestricted valid resolved language survives—there is no semantic regex.
sentence = "Bilateral pleural effusion observed in the old CT was not detected in the current examination."
row = make_row({"Pleural effusion": "resolved"}, sentence)
output, _, _ = combine(row, make_parsed({"Pleural effusion": {
    "report_change": "resolved",
    "evidence": "observed in the old CT was not detected",
    "temporal_sentence": sentence}}))
item = finding(output, "Pleural effusion")
assert (item["change"], item["direction"], item["label_source"]) == (
    "resolved", "improved", "report_explicit")

# Static current statement + structured new -> weak label + deterministic synthetic text.
static = "There is bilateral pleural effusion."
row = make_row({"Pleural effusion": "new"}, static)
parsed = make_parsed({})
output, _, _ = combine(row, parsed)
item = finding(output, "Pleural effusion")
assert (item["change"], item["direction"], item["label_source"], item["temporal_sentence_source"]) == (
    "new", "worsened", "structured_presence", "synthetic_presence")
assert static not in item["temporal_sentence"]
first = (item["temporal_sentence"], item["synthetic_template_id"])
output2, _, _ = combine(row, parsed)
item2 = finding(output2, "Pleural effusion")
assert first == (item2["temporal_sentence"], item2["synthetic_template_id"])

# Structured resolved fallback.
row = make_row({"Pleural effusion": "resolved"}, "No pleural effusion.")
output, _, _ = combine(row, make_parsed({}))
item = finding(output, "Pleural effusion")
assert (item["direction"], item["label_source"], item["temporal_sentence_source"]) == (
    "improved", "structured_presence", "synthetic_presence")

# Present in both but no explicit change -> unknown, never stable.
row = make_row({"Cardiomegaly": "present_both"}, "Cardiomegaly is present.")
output, _, _ = combine(row, make_parsed({}))
item = finding(output, "Cardiomegaly")
assert (item["change"], item["direction"], item["label_source"], item["temporal_sentence"]) == (
    "no_comparison", "unknown", "unknown", "")

# Stable requires an explicit, verbatim report statement.
sentence = "Cardiomegaly remains unchanged from the prior examination."
row = make_row({"Cardiomegaly": "present_both"}, sentence)
output, _, _ = combine(row, make_parsed({"Cardiomegaly": {
    "report_change": "stable", "evidence": "remains unchanged", "temporal_sentence": sentence}}))
item = finding(output, "Cardiomegaly")
assert (item["direction"], item["label_source"]) == ("stable", "report_explicit")

# Absent both stays outside progression.
output, _, _ = combine(make_row({}, ""), make_parsed({}))
item = finding(output, "Lung nodule")
assert (item["change"], item["direction"], item["label_source"]) == ("absent", "unknown", "absent")

# Claimed explicit but non-verbatim text is rejected to safe fallback.
row = make_row({"Pleural effusion": "present_both"}, "There is pleural effusion.")
fake = "Interval increase in pleural effusion."
output, _, _ = combine(row, make_parsed({"Pleural effusion": {
    "report_change": "worse", "evidence": "Interval increase", "temporal_sentence": fake}}))
item = finding(output, "Pleural effusion")
assert item["direction"] == "unknown" and item["rejection_reason"] == "sentence_not_verbatim"

print("PASS: valid JSON; 11 cells; all code compiles; no semantic regex; no truncation;")
print("PASS: exact context check, generation-cap diagnostic, and full failed output preservation")
print("PASS: LLM prompt contains current report but hides prior/structured signals")
print("PASS: postprocessing applies structured fallback; all 8 behavior tests pass")