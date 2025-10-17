"""Generate a RetinaNet architecture diagram using PlotNeuralNet."""
from __future__ import annotations

import argparse
import shutil
import os
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DOCS))

from plot_neural_net import (  # noqa: E402
    to_begin,
    to_Conv,
    to_SoftMax,
    to_connection,
    to_end,
    to_head,
    to_cor,
    to_input,
    to_generate,
)


def build_architecture(project_path: Path, assets_path: Path, output_dir: Path) -> List[str]:
    """Return the TikZ commands describing the RetinaNet model."""

    image_path = Path(os.path.relpath(assets_path / "cell_counting_result.png", output_dir))
    layers_root = os.path.relpath(project_path, output_dir)
    arch: List[str] = [
        to_head(layers_root),
        to_cor(),
        to_begin(),
        to_input(str(image_path).replace("\\", "/"), to="(-5,0,0)", width=5, height=5, name="input"),
    ]

    backbone_specs = [
        ("stem", "Stem 7x7 Conv", 64, 640, "(0,0,0)", 1.5, 28, 28),
        ("c2", "ResNet C2", 256, 320, "(2,0,0)", 2.5, 24, 24),
        ("c3", "ResNet C3", 512, 160, "(4.5,0,0)", 3, 20, 20),
        ("c4", "ResNet C4", 1024, 80, "(7.5,0,0)", 3.2, 16, 16),
        ("c5", "ResNet C5", 2048, 40, "(10.5,0,0)", 3.4, 12, 12),
    ]

    previous_name = "input"
    for name, caption, n_filters, spatial, position, width, height, depth in backbone_specs:
        arch.append(
            to_Conv(
                name,
                s_filer=spatial,
                n_filer=n_filters,
                to=position,
                width=width,
                height=height,
                depth=depth,
                caption=caption,
            )
        )
        arch.append(to_connection(previous_name, name))
        previous_name = name

    fpn_specs = [
        ("p3", "FPN P3", 256, 80, "(4.5,3,0)"),
        ("p4", "FPN P4", 256, 40, "(7.5,3,0)"),
        ("p5", "FPN P5", 256, 20, "(10.5,3,0)"),
        ("p6", "FPN P6", 256, 10, "(13.5,3,0)"),
        ("p7", "FPN P7", 256, 5, "(16.5,3,0)"),
    ]

    for idx, (name, caption, n_filters, spatial, position) in enumerate(fpn_specs):
        arch.append(
            to_Conv(
                name,
                s_filer=spatial,
                n_filer=n_filters,
                to=position,
                width=1.2,
                height=max(10 - idx, 4),
                depth=max(10 - idx, 4),
                caption=caption,
            )
        )

    # Connections from backbone to FPN top-down pathway
    arch.extend(
        [
            to_connection("c3", "p3"),
            to_connection("c4", "p4"),
            to_connection("c5", "p5"),
        ]
    )

    # Heads
    arch.append(
        to_Conv(
            "cls_head",
            s_filer=9,
            n_filer=256,
            to="(19,1.5,0)",
            width=1.2,
            height=9,
            depth=9,
            caption="Classification Subnet",
        )
    )
    arch.append(to_connection("p7", "cls_head"))

    arch.append(
        to_Conv(
            "reg_head",
            s_filer=9,
            n_filer=256,
            to="(19,-1.5,0)",
            width=1.2,
            height=9,
            depth=9,
            caption="Regression Subnet",
        )
    )
    arch.append(to_connection("p7", "reg_head"))

    arch.append(to_SoftMax("detections", s_filer=2, to="(22,1.5,0)", width=0.7, height=6, depth=6, caption="Detections"))
    arch.append(to_connection("cls_head", "detections"))

    arch.append(to_end())
    return arch


def run_pdflatex(tex_path: Path) -> None:
    """Compile the TikZ file with pdflatex if available."""

    if shutil.which("pdflatex") is None:
        print("pdflatex not found; skipping PDF compilation.")
        return

    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={tex_path.parent}",
            str(tex_path),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Directory where the generated TeX (and optional PDF) files will be stored.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the generated TeX file with pdflatex if it is available.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    project_path = Path(__file__).resolve().parents[1] / "plot_neural_net"
    assets_path = Path(__file__).resolve().parents[1] / "assets"

    architecture = build_architecture(project_path, assets_path, output_dir)

    tex_path = output_dir / "retinanet_architecture.tex"
    to_generate(architecture, str(tex_path))

    if args.compile:
        run_pdflatex(tex_path)


if __name__ == "__main__":
    main()
