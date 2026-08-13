from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
WORK_DIR = PROJECT_DIR / "pptx_work"
CREATE_MJS = WORK_DIR / "create_presentation.mjs"
PPTX_OUT = PROJECT_DIR / "presentation.pptx"
TEX_OUT = PROJECT_DIR / "presentation.tex"
PDF_OUT = PROJECT_DIR / "presentation.pdf"


def find_node() -> str:
    node = shutil.which("node")
    if node:
        return node
    user_home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    bundled = user_home / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node.exe"
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError("Node.js was not found. Install Node.js or run from the Codex runtime environment.")


def setup_artifact_workspace(node: str) -> None:
    user_home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    helper = user_home / ".codex" / "plugins" / "cache" / "openai-primary-runtime" / "presentations" / "26.727.11326" / "skills" / "presentations" / "container_tools" / "setup_artifact_tool_workspace.mjs"
    if not helper.exists():
        # The workspace may already be prepared. Keep going and let Node report a clear import error if not.
        return
    env = os.environ.copy()
    env["HOME"] = str(user_home)
    subprocess.run([node, str(helper), "--workspace", str(WORK_DIR)], cwd=PROJECT_DIR, env=env, check=True)


def build_pptx() -> None:
    node = find_node()
    setup_artifact_workspace(node)
    subprocess.run([node, str(CREATE_MJS)], cwd=WORK_DIR, check=True)


def build_pdf() -> None:
    xelatex = shutil.which("xelatex")
    if not xelatex:
        print("XeLaTeX was not found, so presentation.pdf was not rebuilt.", file=sys.stderr)
        return
    for _ in range(2):
        subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", TEX_OUT.name],
            cwd=PROJECT_DIR,
            check=True,
        )


def main() -> None:
    if not CREATE_MJS.exists():
        raise FileNotFoundError(f"Missing slide generator: {CREATE_MJS}")
    if not TEX_OUT.exists():
        raise FileNotFoundError(f"Missing LaTeX source: {TEX_OUT}")
    build_pptx()
    build_pdf()
    print(f"Created: {PPTX_OUT}")
    if PDF_OUT.exists():
        print(f"Created: {PDF_OUT}")
    print(f"Source:  {TEX_OUT}")


if __name__ == "__main__":
    main()
