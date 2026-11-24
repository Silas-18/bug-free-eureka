#!/usr/bin/env python3
"""Create separate LLVIP structure-ablation model/train/eval scripts.

Run from the root directory of LLVIP_Project. Original scripts are not changed.
Modes:
  concat : original concat -> 1x1 conv -> BN -> SiLU (the current method)
  avg    : elementwise (RGB + IR) / 2 at P3/P4/P5 (zero trainable fusion parameters)

No GPU or installed ML dependencies are required to run this preparer.
"""
from pathlib import Path
import re
import ast
import copy

ROOT = Path.cwd()
CODE = ROOT / "code"
ORIGINALS = {
    "model": CODE / "fusion_model.py",
    "train": CODE / "train_fusion_stage1.py",
    "eval": CODE / "eval_fusion.py",
}
OUTPUTS = {
    "model": CODE / "fusion_model_structure.py",
    "train": CODE / "train_fusion_structure.py",
    "eval": CODE / "eval_fusion_structure.py",
}


def exactly_once(s, old, new, label):
    count = s.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly 1 occurrence, found {count}. No files changed.")
    return s.replace(old, new, 1)


def prepare_model(s):
    s = exactly_once(
        s,
        "def __init__(self, rgb_weights, ir_weights):",
        "def __init__(self, rgb_weights, ir_weights, fusion_mode='concat'):",
        "model constructor",
    )
    s = exactly_once(
        s,
        "        super().__init__()",
        "        super().__init__()\n"
        "        if fusion_mode not in ('concat', 'avg'):\n"
        "            raise ValueError(f'Unknown fusion_mode: {fusion_mode}')\n"
        "        self.fusion_mode = fusion_mode",
        "model mode validation",
    )
    marker = '        print("双模态融合模型初始化完成！")'
    s = exactly_once(
        s,
        marker,
        "        # avg requires no learned fusion module; keep the .fusions attribute\n"
        "        # for compatibility with the existing parameter-freezing code.\n"
        "        if self.fusion_mode == 'avg':\n"
        "            self.fusions = nn.ModuleList([nn.Identity() for _ in channels])\n\n"
        "        print(f'融合方式: {self.fusion_mode}')\n" + marker,
        "model fusions",
    )
    # Replace only the 3-scale fusion loop, retaining downstream YOLO-neck routing.
    pattern = re.compile(
        r"(?m)^        for i in range\(3\):\n"
        r"(?P<body>(?:.*\n)*?)"
        r"(?=^        # 构建 YOLO Neck 所需的特征索引)",
    )
    matches = list(pattern.finditer(s))
    if len(matches) != 1:
        raise RuntimeError(f"model fusion loop: expected 1, found {len(matches)}. No files changed.")
    if "self.fusions[i]" not in matches[0].group("body") or "torch.cat" not in matches[0].group("body"):
        raise RuntimeError("Model fusion body was not recognized; no files changed.")
    replacement = (
        "        for i in range(3):\n"
        "            if self.fusion_mode == 'avg':\n"
        "                # Same geometry and channel count as the original features.\n"
        "                feature = (rgb_features[i] + ir_features[i]) * 0.5\n"
        "            else:\n"
        "                feature = torch.cat([rgb_features[i], ir_features[i]], dim=1)\n"
        "                feature = self.fusions[i](feature)\n"
        "            fused_features.append(feature)\n\n"
    )
    return s[:matches[0].start()] + replacement + s[matches[0].end():]


def prepare_script(s, which):
    """Modify one original training/evaluation script without assuming its CLI layout."""
    s = exactly_once(
        s,
        "from fusion_model import DualModalYOLOv8",
        "from fusion_model_structure import DualModalYOLOv8",
        f"{which} import",
    )

    # Some versions use `return parser.parse_args()`, others use an
    # assignment (`args = parser.parse_args()`) or a different parser name.
    # Inspect the actual Python syntax instead of looking for an exact string.
    tree = ast.parse(s)
    parse_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "parse_args"
        and isinstance(n.func.value, ast.Name)
    ]
    if len(parse_calls) != 1:
        raise RuntimeError(
            f"{which} CLI: expected 1 parser.parse_args() call, "
            f"found {len(parse_calls)}. No files changed."
        )
    call = parse_calls[0]
    parser_var = call.func.value.id

    # Confirm this name is an argparse parser rather than an unrelated API.
    parser_has_args = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == parser_var
        for n in ast.walk(tree)
    )
    if not parser_has_args:
        raise RuntimeError(
            f"{which} CLI: cannot verify '{parser_var}' is the argument parser. "
            "No files changed."
        )

    lines = s.splitlines(keepends=True)
    line = lines[call.lineno - 1]
    indent = line[: len(line) - len(line.lstrip())]
    new_cli = (
        f"{indent}{parser_var}.add_argument('--fusion-mode', "
        "choices=['concat', 'avg'], default='concat', "
        "help='concat: original CBR; avg: elementwise mean')\n"
    )
    lines.insert(call.lineno - 1, new_cli)
    s = "".join(lines)

    # Replace just the constructor expression. AST allows both positional
    # and named arguments, comments/newlines and different argument spacing.
    tree = ast.parse(s)
    constructors = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "DualModalYOLOv8"
    ]
    if len(constructors) != 1:
        raise RuntimeError(
            f"{which} model initialization: expected 1 DualModalYOLOv8 call, "
            f"found {len(constructors)}. No files changed."
        )
    node = constructors[0]
    if any(kw.arg == "fusion_mode" for kw in node.keywords):
        raise RuntimeError(f"{which}: fusion_mode already configured. No files changed.")

    # Use `args.fusion_mode`, matching the CLI argument convention of the
    # project's existing train and eval scripts.
    updated_node = copy.deepcopy(node)
    updated_node.keywords.append(
        ast.keyword(
            arg="fusion_mode",
            value=ast.Attribute(
                value=ast.Name(id="args", ctx=ast.Load()),
                attr="fusion_mode", ctx=ast.Load(),
            ),
        )
    )
    original_call = ast.get_source_segment(s, node)
    if original_call is None or s.count(original_call) != 1:
        raise RuntimeError(f"{which}: cannot locate model constructor text. No files changed.")
    s = s.replace(original_call, ast.unparse(updated_node), 1)
    return s


def main():
    if not CODE.is_dir() or any(not p.is_file() for p in ORIGINALS.values()):
        raise SystemExit(
            "Please run at /root/autodl-tmp/LLVIP_Project, "
            "with original model/train/eval scripts in code/."
        )
    existing = [str(p) for p in OUTPUTS.values() if p.exists()]
    if existing:
        raise SystemExit("Refusing to overwrite existing files: " + ", ".join(existing))

    original = {k: p.read_text(encoding="utf-8") for k, p in ORIGINALS.items()}
    generated = {
        "model": prepare_model(original["model"]),
        "train": prepare_script(original["train"], "train"),
        "eval": prepare_script(original["eval"], "eval"),
    }
    for k, source in generated.items():
        compile(source, str(OUTPUTS[k]), "exec")

    for k, target in OUTPUTS.items():
        target.write_text(generated[k], encoding="utf-8")
        print("Created:", target.relative_to(ROOT))

    print("\nOriginal files were not modified.")
    print("Use --fusion-mode avg for the mean-fusion comparison, and concat for the original fusion.")
    print("Do not use --init-weights from stage1 for this controlled comparison.")


if __name__ == "__main__":
    main()
