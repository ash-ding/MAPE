"""Assemble self-contained paper harnesses without invoking models."""
import argparse
import copy
import shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "programs/mape_a"
EXPERIMENTS = ROOT / "experiments"
BASELINES = ("amem", "lightmem", "mem0", "simplemem", "zep")

def copy_method(dest, entry="mape_a.py"):
    shutil.copyfile(METHOD / "memo.py", dest / entry)
    shutil.copytree(METHOD / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__"))

def prepare(output):
    output = Path(output).resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must be inside the repository")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    templates = {b: yaml.safe_load((ROOT / "configs/evaluation" / f"mape_a_{b}.yaml").read_text())
                 for b in ("locomo", "longmemeval")}
    recipes = []
    # Direct evaluation of the sole released final program.
    for bench in templates:
        name = f"{bench}_method"
        dest = output / name / "harness"
        dest.mkdir(parents=True)
        copy_method(dest, "memo.py" if bench == "longmemeval" else "memo_lme.py")
        if bench == "locomo":
            shutil.copyfile(EXPERIMENTS / "transfer/locomo.py", dest / "memo.py")
            shutil.copyfile(METHOD / "adapters/locomo.py", dest / "locomo_adapter.py")
        recipes.append((name, bench, dest, None))
    # These historical ablations share the memory cache created by their full arm.
    for bench in templates:
        for arm in ("full", "noraw", "noraw_nocompose"):
            name = f"{bench}_{arm}"
            dest = output / name / "harness"
            source = EXPERIMENTS / "ablation" / bench / arm
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__"))
            copy_method(dest)
            if bench == "locomo":
                shutil.copyfile(METHOD / "adapters/locomo.py", dest / "locomo_adapter.py")
            recipes.append((name, bench, dest, None))
    # Replay arms: retrieval fixed from a recorded run (replay.json, built by
    # scripts/build_replay.py), only the compose stage varies.
    replay_arms = [("heldout_lme_qwen_acc_7723a2_nocompose", "longmemeval", None),
                   ("heldout_locomo_qwen_a20_7723a2_nocompose", "locomo", None)]
    for baseline in BASELINES:
        replay_arms.append((f"heldout_lme_qwen_control_{baseline}", "longmemeval", None))
        replay_arms.append((f"heldout_lme_qwen_transplant_{baseline}", "longmemeval", baseline))
    for name, bench, baseline in replay_arms:
        dest = output / name / "harness"
        dest.mkdir(parents=True)
        if baseline:
            shutil.copyfile(EXPERIMENTS / "transplantation/memo.py", dest / "memo.py")
            shutil.copyfile(EXPERIMENTS / "transplantation/adapters" / f"{baseline}.py", dest / "adapters.py")
            copy_method(dest)
        else:
            shutil.copyfile(EXPERIMENTS / "ablation/replay.py", dest / "memo.py")
        recipes.append((name, bench, dest, None))
    for name, bench, dest, settings in recipes:
        cfg = copy.deepcopy(templates[bench])
        cfg["run_name"] = "paper_" + name
        cfg["harnesses"] = [str(dest.relative_to(ROOT))]
        if name.endswith("nocompose") or "_control_" in name:
            cfg["contract"]["compose"] = False
        (dest.parent / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return [output / name / "config.yaml" for name, _, _, _ in recipes]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/paper")
    args = parser.parse_args()
    paths = prepare(args.output)
    print(f"Prepared {len(paths)} experiments under {args.output}; no models called.")
    print("Replay experiments require replay.json: run the full arm (or the baseline) first, then scripts/build_replay.py.")
