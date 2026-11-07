"""Locally package completed C++1031 results; never run models or draw figures."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024 * 1024
BITS = (np.arange(16)[:, None] >> np.arange(4)) & 1


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def copy_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected regular source file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, CHUNK)


def _stream_hash(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(CHUNK), b""):
        h.update(block)
    return h.hexdigest()


def compress_table(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as out:
            shutil.copyfileobj(src, out, CHUNK)
    expected = sha_file(source)
    with gzip.open(target, "rb") as check:
        if _stream_hash(check) != expected:
            raise ValueError(f"Compressed CSV payload changed: {source}")
    return dict(source_sha256=expected, packaged_sha256=sha_file(target),
                source_bytes=source.stat().st_size, packaged_bytes=target.stat().st_size)


def repack_npz(source: Path, target: Path) -> dict:
    """Change ZIP compression only; check every exact NPY member after writing."""
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if (not names or len(set(names)) != len(names)
                or any(Path(n).name != n or "\\" in n or not n.endswith(".npy") for n in names)):
            raise ValueError(f"Unsafe/non-NPY members: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        members = {}
        try:
            with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as out:
                for name in names:
                    h = hashlib.sha256()
                    with z.open(name) as src, out.open(name, "w", force_zip64=True) as dst:
                        first = src.read(CHUNK)
                        if not first.startswith(b"\x93NUMPY"):
                            raise ValueError(f"Invalid NPY payload: {name}")
                        h.update(first)
                        dst.write(first)
                        for block in iter(lambda: src.read(CHUNK), b""):
                            h.update(block)
                            dst.write(block)
                    members[name] = h.hexdigest()
            with zipfile.ZipFile(target) as check:
                if check.namelist() != names:
                    raise ValueError("NPZ members changed during compression")
                for name, digest in members.items():
                    with check.open(name) as stream:
                        if _stream_hash(stream) != digest:
                            raise ValueError(f"Repacked payload differs: {target}/{name}")
        except BaseException:
            target.unlink(missing_ok=True)  # Only this invocation's fresh output.
            raise
    return dict(member_count=len(members), npy_sha256=members,
                source_sha256=sha_file(source), packaged_sha256=sha_file(target),
                source_bytes=source.stat().st_size, packaged_bytes=target.stat().st_size)


def rule_masks(catalogue: dict) -> np.ndarray:
    masks = []
    for expected, rule in enumerate(catalogue["rules"]):
        if rule["index"] != expected:
            raise ValueError("Catalogue rule identity/order mismatch")
        used = np.zeros(4, dtype=bool)
        for category in rule["regions"]:
            for component in category:
                a = np.asarray(component["A"], float)
                if a.size:
                    used |= (np.abs(a) > 1e-12).any(axis=0)
        masks.append(int(used @ (1 << np.arange(4))))
    return np.asarray(masks, dtype=np.int64)


def filtered_execution(arrays: dict, masks: np.ndarray) -> np.ndarray:
    """Exact linear reconstruction across ALL PF seeds, not a seed-0 estimate."""
    sets = np.asarray(arrays["conditional_feature_set_probability"])
    execution = np.asarray(arrays["conditional_catalogue_execution_probability"])
    valid = np.asarray(arrays["valid"], bool)
    n = len(valid)
    if sets.shape != (n, 16) or execution.shape != (n, len(masks)) or np.any((masks < 1) | (masks > 15)):
        raise ValueError("Invalid feature/catalogue axes; rules cannot be uncommitted")
    if (not np.isfinite(sets[valid]).all() or not np.isfinite(execution[valid]).all()
            or (sets[valid] < 0).any() or (execution[valid] < 0).any()
            or not np.allclose(sets[valid].sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Invalid feature/execution probabilities")
    if not np.isnan(sets[~valid]).all() or not np.isnan(execution[~valid]).all():
        raise ValueError("Missed trials must remain NaN")
    rule_sets = np.zeros((n, 16))
    for code in range(1, 16):
        rule_sets[:, code] = execution[:, masks == code].sum(axis=1)
    residual = sets - rule_sets
    if np.any(residual[valid] < -1e-10):
        raise ValueError("Rule execution exceeds matching feature-set probability")
    # Subtraction can leave roundoff-sized negative zeros, never real mass.
    residual[valid] = np.maximum(residual[valid], 0)
    result = np.column_stack([residual[:, 0], residual[:, 15],
                              residual[:, 1:15].sum(axis=1), execution.sum(axis=1)])
    result[~valid] = np.nan
    if not np.allclose(result[valid].sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("Reconstructed execution does not normalize")
    return result


def python_fit_arrays(a: dict, masks: np.ndarray) -> dict:
    valid = np.asarray(a["valid"], bool)
    forecast = a["conditional_category_probabilities"]
    truth = a["true_categories"]
    correct = np.full(len(valid), np.nan)
    rows = np.flatnonzero(valid)
    correct[rows] = forecast[rows, truth[rows]]
    sets = a["conditional_feature_set_probability"]
    out = dict(choice_probabilities=forecast,
        choice_probabilities_by_repeat=a["conditional_category_probability_repeats"],
        feature_set_probability=sets, feature_probability=sets @ BITS,
        catalogue_execution_probability=a["conditional_catalogue_execution_probability"],
        representation_belief=a["conditional_prior"],
        representation_execution=a["conditional_choice_weights"],
        filtered_representation_belief=a["conditional_filtered_prior"],
        filtered_representation_execution=filtered_execution(a, masks),
        stimulus=a["stimuli"], choices=a["observed_choices"] + 1,
        categories=truth + 1, feedback=a["observed_feedback"],
        valid=a["valid"], score_mask=a["score_mask"],
        trial_ids=a["trial_ids"], session_ids=a["session_ids"],
        pf_seeds=a["pf_seeds"], conditional_correct_probability=correct)
    for key in ("catalogue_rule_ids", "conditional_choice_probabilities", "conditional_score_mask"):
        if key in a:
            out[key] = a[key]
    for name in ("catalogue_prior", "catalogue_filtered_prior", "catalogue_posterior",
                 "catalogue_predictive_execution_probability", "catalogue_active_probability"):
        if "conditional_"+name in a:
            out[name] = a["conditional_"+name]
    return out


def python_generation_arrays(a: dict) -> dict:
    valid = np.asarray(a["autonomous_valid"], bool)
    correct = np.where(valid, a["autonomous_choices"] == a["true_categories"][None], np.nan)
    return dict(choices=a["autonomous_choices"]+1, feedback=a["autonomous_feedback"],
        valid=a["autonomous_valid"], category_probabilities=a["autonomous_category_probabilities"],
        categories=a["true_categories"]+1, choice_correct=correct,
        seeds=a["simulation_seeds"], trial_ids=a["trial_ids"], session_ids=a["session_ids"])


def save_arrays(path: Path, arrays: dict) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as check:
        if set(check.files) != set(arrays):
            raise ValueError(f"Derived NPZ keys changed: {path}")
        for key, reference in arrays.items():
            actual = check[key]
            if actual.shape != reference.shape or actual.dtype != reference.dtype or actual.tobytes() != reference.tobytes():
                raise ValueError(f"Derived NPZ round trip differs: {key}")


def archive_tree(source: Path, target: Path) -> dict:
    files = sorted(p for p in source.rglob("*") if p.is_file())
    proof = []
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as out:
        for file in files:
            if file.is_symlink() or file.suffix.lower() == ".png":
                raise ValueError(f"Unexpected fit artifact: {file}")
            name = file.relative_to(source).as_posix()
            out.write(file, name)
            proof.append(dict(path=name, sha256=sha_file(file), bytes=file.stat().st_size))
    with zipfile.ZipFile(target) as check:
        for item in proof:
            with check.open(item["path"]) as stream:
                if _stream_hash(stream) != item["sha256"]:
                    raise ValueError(f"Fit archive payload differs: {item['path']}")
    return dict(source_file_count=len(files), files=proof)


def completed_results(futures):
    try:
        for future in as_completed(futures):
            yield future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        raise


def subject_rows(rows: list[dict], arrays: dict) -> list[dict]:
    lookup = {}
    for row in rows:
        key = tuple(int(row[k]) for k in ("iSession", "iBlock", "iTrial"))
        if key in lookup:
            raise ValueError("Duplicate source trial ID")
        lookup[key] = row
    ids = arrays["trial_ids"]
    selected = [lookup[tuple(int(x) for x in key)] for key in ids]
    if len(selected) != len(rows):
        raise ValueError("Full subject source does not match saved trial count")
    expected = np.array([[float(row[f"feature{d}"]) for d in range(1, 5)] for row in selected])
    if not np.array_equal(expected, arrays["stimuli"]):
        raise ValueError("Saved physical stimuli differ from source CSV")
    for t, row in enumerate(selected):
        valid = int(row.get("miss", "0").strip() or "0") == 0
        if valid != bool(arrays["valid"][t]):
            raise ValueError("Source miss mask differs")
        if valid and (int(row["choice"])-1 != arrays["observed_choices"][t]
                      or float(row["feedback"]) != arrays["observed_feedback"][t]):
            raise ValueError("Observed choice/feedback differs from original CSV")
        if int(row["category"])-1 != arrays["true_categories"][t]:
            raise ValueError("Category coordinate mismatch")
    return selected


def pack_subject(base: Path, destination: Path, source_rows: list[dict], masks: np.ndarray) -> dict:
    sid = int(base.name.removeprefix("subject_"))
    evaluation = base / "evaluate" / f"subject_{sid}"
    fit = base / "fit" / f"subject_{sid}"
    m = read_json(evaluation / "evaluation_manifest.json")
    best = read_json(fit / "best_parameters.json")
    stats = read_json(base / "png/summary.json")
    if (m["subject_id"] != sid or m["parameters"] != best["best_parameters"]
            or m["data_fingerprint"] != best["data_fingerprint"]
            or m["provenance"] != best["provenance"]):
        raise ValueError(f"Fit/evaluation identity mismatch: {sid}")
    packet = m["conditional_particle_contents"]
    if packet["status"] != "available" or packet["phase"] != "current_choice_filtered_before_feedback_before_resampling":
        raise ValueError(f"Conditional particle contents missing/wrong phase: {sid}")
    destination.mkdir(parents=True, exist_ok=False)
    native = destination / "native"
    native.mkdir()
    proofs = {}
    for name, expected in (("evaluation_inputs.npz", m["inputs_sha256"]),
                           ("conditional_particle_contents.npz", packet["sha256"])):
        if name != "evaluation_inputs.npz" and packet["file"] != name:
            raise ValueError("Unexpected particle archive filename")
        proof = repack_npz(evaluation / name, native / name)
        if proof["source_sha256"] != expected:
            raise ValueError(f"Original archive checksum failed: {sid}/{name}")
        proofs[name] = proof
    # Keep both original identities and a portable manifest for repacked archives.
    copy_file(evaluation / "evaluation_manifest.json", native / "evaluation_manifest.original.json")
    repacked = json.loads(json.dumps(m))
    repacked["inputs_sha256"] = proofs["evaluation_inputs.npz"]["packaged_sha256"]
    repacked["conditional_particle_contents"]["sha256"] = proofs["conditional_particle_contents.npz"]["packaged_sha256"]
    repacked["packaging"] = dict(operation="lossless ZIP recompression; identical NPY member bytes",
        original_manifest="evaluation_manifest.original.json", original_inputs_sha256=m["inputs_sha256"],
        original_particle_sha256=packet["sha256"])
    write_json(native / "evaluation_manifest.json", repacked)
    copy_file(evaluation / "content_path.json", native / "content_path.json")
    with np.load(native / "evaluation_inputs.npz", allow_pickle=False) as z:
        a = {name: z[name] for name in z.files}
    n = len(a["valid"])
    if n != m["trials"] or a["conditional_catalogue_execution_probability"].shape != (n, len(masks)):
        raise ValueError("Trial/catalogue axes disagree")
    pyfit, pygen = python_fit_arrays(a, masks), python_generation_arrays(a)
    save_arrays(destination / "python1031_fit.npz", pyfit)
    save_arrays(destination / "python1031_generation.npz", pygen)
    selected = subject_rows(source_rows, a)
    with (destination / "subject_source.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    copy_file(fit / "best_parameters.json", destination / "best_parameters.json")
    copy_file(fit / "fit_manifest.json", destination / "fit_manifest.json")
    fit_proof = archive_tree(fit, destination / "fit_history.zip")
    table_proofs = {}
    for file in (base / "paper").iterdir():
        if file.is_file() and file.suffix.lower() != ".png":
            if file.suffix.lower() == ".csv":
                table_proofs[file.name] = compress_table(file, destination / "tables" / (file.name+".gz"))
            else:
                copy_file(file, destination / "tables" / file.name)
    (destination / "plotting").mkdir()
    plot_proof = repack_npz(base / "png/arrays.npz", destination / "plotting/arrays.npz")
    copy_file(base / "png/summary.json", destination / "plotting/summary.original.json")
    meta = dict(schema_version="gm1031-python-handoff-subject-v1", subject_id=sid,
        condition=m["condition"], n_categories=m["n_categories"], trials=n,
        parameters=m["parameters"], feature_names=m["protocol"]["feature_order"], protocol=m["protocol"],
        conditional_view="equal mean of all PF repeats; current-choice filtered latent arrays before current feedback",
        predictive_view="choice_probabilities and representation_* use history before current choice",
        conditional_particle_file="native/conditional_particle_contents.npz",
        conditional_particle_contents=repacked["conditional_particle_contents"],
        pf_repeats=len(m["pf_seeds"]), autonomous_repeats=len(m["simulation_seeds"]),
        coordinate="Python choices/categories 1..C, missing choices=0; native files retain 0..C-1 and missing=-1",
        filtered_execution_derivation="feature-set execution minus catalogue execution grouped by rule feature support; uncommitted bin0, full-example bin15, generalized bins1:15, catalogue sum",
        unavailable=["seed-wise latent summaries for all PF repeats", "internal states of autonomous paths 1..511", "all-repeat ESS/resampling arrays", "geometric oral IoU"],
        original_provenance=m["provenance"], data_fingerprint=m["data_fingerprint"],
        rule_count=len(masks), resource_fingerprint=m["resource_fingerprint"])
    write_json(destination / "python1031_metadata.json", meta)
    write_json(destination / "preservation_checks.json", dict(npz=proofs, plotting_npz=plot_proof,
        fit_archive=fit_proof, tables=table_proofs, source_subject=str(base), npy_members_identical=True,
        physical_stimuli_choices_feedback_trial_ids_match_source_csv=True))
    return dict(subject_id=sid, condition=m["condition"], trials=n,
        valid_trials=int(a["valid"].sum()), scored_trials=int(a["conditional_score_mask"].sum()),
        pf_repeats=len(m["pf_seeds"]), particles=m["config"]["particle_count"],
        autonomous_repeats=len(m["simulation_seeds"]), rule_count=len(masks),
        fit_mean_choice_nll=best["mean_nll"],
        evaluation_mean_choice_nll=stats["conditional"]["mean_choice_nll"],
        evaluation_mean_choice_brier=stats["conditional"]["mean_response_brier"],
        **m["parameters"])


README = """# 1031 C++ 三条件全量结果交付

本目录包含 Cond1/2/3 各32名、共96名被试的全部已有数值结果，不包含PNG图像。
没有重新拟合、重新模拟或改变原始结果；大型NPZ只改ZIP压缩，每个内部NPY字节均已校验相同。
原始目录未修改。请将整个目录一起交付，不要只发送均值表。
目录包含被试行为和原始口述资料，仅供获授权的合作者使用；本工具不自动上传任何内容。

## 从哪里开始

- `subjects.csv`：逐被试参数、trial数、拟合与最终evaluation的NLL等；两种NLL预算不同，勿混用。
- `subjects/cond1/subject_101/`：每名被试一个目录，其他条件同理。
- `python1031_fit.npz`：适配Python1031常用字段的预测/条件状态数据。
- `python1031_generation.npz`：全部512条自主路径，choice正确率与graded feedback分开保存。
- `native/`：完整原生evaluation及真实历史、反馈前的逐粒子内容；含H0/H4/H6及全部目录规则。
- `tables/`：逐trial/窗口规则表、部位使用和已有口述编码比较表；CSV以`.csv.gz`无损保存，Pandas可直接读取。
- `plotting/arrays.npz`：原已生成绘图的全部数值输入和bands；不需要重新运行模型。
- `fit_history.zip`：完整优化历史与缓存，文件内容逐项校验，必要时可解压检查。
- `subject_source.csv`：与保存trial顺序逐条核对的原始被试数据行，含原始口述/维度名称等。
- `resources/`：目录规则几何和标签；H编号必须结合各被试的feature_order解释。
- `run_metadata/`：运行日志、启动信息及配置快照。
- `source_provenance/`：当前1031数据来源声明及本次打包程序，供追溯，不是额外模拟结果。

## Python读取（无需原模型代码）

```python
from python1031_io import load_subject, load_particles
data = load_subject('.', 1, 101)
fit, generation = data['fit'], data['generation']
p_h0 = fit['catalogue_execution_probability'][:, 0]
p_h4 = fit['catalogue_execution_probability'][:, 4]
p_h6 = fit['catalogue_execution_probability'][:, 6]
groups = fit['filtered_representation_execution']
with load_particles('.', 1, 101) as particles:
    weights = particles['weights']
    beta = particles['beta']
    anchors = particles['template_anchor']
```

接口仅依赖NumPy，内容数组按需读取，不要同时加载96名被试的全部粒子。
运行 `python python1031_io.py --verify .` 可检查交付目录中的文件哈希。
具体返回格式见 `FIELD_MAP.md`。这不是旧版multi-model pilot研究目录的伪造副本；
旧研究脚本仍需修改路径/选择逻辑，不能未经改动直接运行其registration/ranking流程。

## 必须注意的语义

1. `choice_probabilities`为当前choice之前的预测；`filtered_*`与粒子内容为当前choice已知、当前feedback未知。
2. 条件均值使用全部16次PF；完整粒子快照只来自事先固定的第一个PF seed，不是最优seed。
3. `representation_execution`不是`filtered_representation_execution`；后者用已有完整分布精确线性还原，未用前者顶替。
   浮点相减产生的绝对值不超过1e-10的负残差置零，不重新归一化；本批次实际改变量至多约5.7e-16。
4. Python兼容数据的choice/category为1-based；原生NPZ保持0-based，miss必须按valid排除。
5. Cond3的feedback可为0.5，不等于choice正确率。标签信念保留pairing条件维度，不能随意展平。
6. 工作槽/uid只在单粒子内有意义，不是跨粒子的同一个规则；稳定规则ID另存。
7. 多数内部状态只有跨PF均值；自主内部状态只保存固定path0，不能伪造512条内部状态或数值重复区间。
8. 口述为外部分析资料，未进入拟合。维度集合吻合/Jaccard不等于几何区域IoU；几何分析可使用条件粒子与规则几何另行实现。
9. 该拟合使用全记录选参；逐trial过滤不使用未来反馈，但参数选择本身不是留出验证，也不是完整后验平滑。
10. `resources/source_manifest.json`是旧资源导出的历史声明，其中旧CSV哈希不能替代当前
    `source_provenance/source_1031.json`；有效感知参数已逐人保存在原生manifest中。
11. 首个有效试次保留在轨迹并参与学习，但不计入native NLL。比较指标必须使用保存的score_mask；
    对PF预测概率先求均值再取log，不等于把各PF的NLL取均值，也不同于旧pilot对所有题的NLL求和。

`manifest.json`列出交付文件及SHA256；每名被试的`preservation_checks.json`记录原始与压缩后哈希、
每个NPY成员的字节一致性和优化缓存成员校验。完成运行约3小时26分钟，8个计算worker。
"""


def package(input_dir: Path, output: Path, jobs: int = 2) -> dict:
    input_dir, output = input_dir.resolve(), output.absolute()
    if output.exists() or output.is_symlink() or input_dir == output or input_dir in output.parents:
        raise ValueError("Output must be a new directory outside the original batch")
    if not 1 <= jobs <= 2:
        raise ValueError("Packaging supports one or two bounded-memory workers")
    launch = read_json(input_dir.parent / "launch.json")
    if launch.get("status") != "completed" or launch.get("exit_code") != 0:
        raise ValueError("Source batch did not complete successfully")
    bases = sorted(input_dir.glob("cond*/subject_*"))
    expected = {c*100+s for c in (1, 2, 3) for s in range(1, 33)}
    if {int(p.name.removeprefix("subject_")) for p in bases} != expected or len(bases) != 96:
        raise ValueError("Expected exactly the 96 condition1/2/3 subjects")
    catalogues = {c: read_json(ROOT / f"resources/catalogue_c{1 if c==1 else 2}.json") for c in (1, 2, 3)}
    masks = {c: rule_masks(cat) for c, cat in catalogues.items()}
    rows = {sid: [] for sid in expected}
    source_csv = ROOT.parents[1] / "data/exp123/processed/Task2_processed.csv"
    with source_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["iSub"]) in rows:
                rows[int(row["iSub"])].append(row)
    for base in bases:
        sid = int(base.name.removeprefix("subject_"))
        ev = base / "evaluate" / base.name
        for path in (ev/"evaluation_inputs.npz", ev/"evaluation_manifest.json", ev/"conditional_particle_contents.npz",
                     base/"fit"/base.name/"best_parameters.json", base/"png/arrays.npz", base/"paper/rule_trials.csv"):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing/unsafe source result: {path}")
        m = read_json(ev / "evaluation_manifest.json")
        if (m["subject_id"] != sid or m["condition"] != sid//100
                or base.parent.name != f"cond{sid//100}" or len(rows[sid]) != m["trials"]):
            raise ValueError(f"Subject/condition/full-trial mismatch: {sid}")
        code = 1 if sid//100==1 else 2
        resource = hashlib.sha256((sha_file(ROOT/f"resources/catalogue_c{code}.json")
                                   +sha_file(ROOT/f"resources/catalogue_c{code}.npz")).encode()).hexdigest()
        if m["resource_fingerprint"] != resource:
            raise ValueError("Current catalogue differs from fitted resource")
    if shutil.disk_usage(output.parent).free < 55 * 2**30:
        raise ValueError("Need 55 GiB free before lossless packaging")
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "packaging_progress.jsonl"
    records = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}
        for base in bases:
            sid = int(base.name.removeprefix("subject_"))
            destination = output / "subjects" / base.parent.name / base.name
            futures[executor.submit(pack_subject, base, destination, rows[sid], masks[sid//100])] = sid
        for record in completed_results(futures):
            records.append(record)
            event = dict(subject=record["subject_id"], completed=len(records), total=96, time=time.time())
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event)+"\n")
            print(json.dumps(event), flush=True)
    records.sort(key=lambda r:r["subject_id"])
    with (output/"subjects.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for file in (ROOT/"resources").iterdir():
        if file.is_file():
            copy_file(file, output/"resources"/file.name)
    copy_file(ROOT/"resources/data_sources.json", output/"source_provenance/source_1031.json")
    copy_file(Path(__file__), output/"source_provenance/package_results_1031.py")
    for source in input_dir.parent.iterdir():
        if source.is_file() and source.suffix.lower() in (".json", ".log"):
            copy_file(source, output/"run_metadata"/source.name)
    for source in (input_dir.parent/"config_snapshots").glob("*"):
        copy_file(source, output/"run_metadata/config_snapshots"/source.name)
    for name in ("plan.json", "progress.log"):
        copy_file(input_dir/name, output/"run_metadata/batch"/name)
    for source in (input_dir/"logs").glob("*"):
        copy_file(source, output/"run_metadata/batch/logs"/source.name)
    copy_file(ROOT/"evaluation/collaboration_io.py", output/"python1031_io.py")
    copy_file(ROOT/"docs/collaboration_field_map.md", output/"FIELD_MAP.md")
    with (output/"README.md").open("x", encoding="utf-8") as stream:
        stream.write(README)
    write_json(output/"validation.json", dict(subjects=96, counts={str(c):32 for c in (1,2,3)},
        all_npz_members_byte_identical=True, all_fit_history_members_sha_verified=True,
        all_subject_csv_trials_verified=True, filtered_execution_validated=True,
        figures_generated=False, source_modified=False, packaging_workers=jobs,
        source_csv_sha256=sha_file(source_csv), unavailable_fields_documented=True))
    files = sorted(p for p in output.rglob("*") if p.is_file())
    inventory = [dict(path=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=sha_file(p)) for p in files]
    manifest = dict(schema_version="gm1031-collaboration-bundle-v1", status="complete",
        subjects=records, files=inventory, file_count=len(inventory),
        bytes=sum(item["bytes"] for item in inventory), source_batch=str(input_dir),
        source_launch=launch, npz_compression="ZIP DEFLATE level1, exact NPY bytes retained", created=time.time())
    write_json(output/"manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name]="1"
    result = package(args.input, args.output, args.jobs)
    print(json.dumps(dict(status=result["status"], subjects=len(result["subjects"]),
        files=result["file_count"], GiB=result["bytes"]/2**30, output=str(args.output))))


if __name__ == "__main__":
    main()
