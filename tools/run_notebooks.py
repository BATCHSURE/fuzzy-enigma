"""Execute every notebook in a fresh kernel; keep source cells unchanged.

Run: python tools/run_notebooks.py [--write] [--kernel-free]
The kernel-free fallback renders plots and captures outputs without localhost
kernel sockets, which is useful in restricted development sandboxes.
"""

import argparse
import base64
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def _kernel_free(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nbformat

    nb = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    images = 0
    count = 0
    start = time.perf_counter()
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        count += 1
        cell.execution_count = count
        cell.outputs = []
        capture = io.StringIO()

        def show(*_args, **_kwargs):
            nonlocal images
            for figure_number in plt.get_fignums():
                fig = plt.figure(figure_number)
                output = io.BytesIO()
                fig.savefig(output, format="png", bbox_inches="tight")
                cell.outputs.append(nbformat.v4.new_output("display_data", data={
                    "image/png": base64.b64encode(output.getvalue()).decode("ascii"),
                    "text/plain": repr(fig),
                }))
                images += 1
            plt.close("all")

        plt.show = show
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                exec(compile(cell.source, f"{path.name}:cell{index}", "exec"), namespace)
        except Exception as exc:
            cell.outputs.append(nbformat.v4.new_output("error", ename=type(exc).__name__,
                                evalue=str(exc), traceback=traceback.format_exc().splitlines()))
            raise RuntimeError(f"{path.name} cell {index} failed") from exc
        if capture.getvalue():
            cell.outputs.insert(0, nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue()))
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    print(f"PASS {path.name}: {count} cells, {images} figures, {time.perf_counter()-start:.2f}s")
    return nb


def _worker(args):
    path = args.notebook.resolve()
    if args.kernel_free:
        nb = _kernel_free(path)
    else:
        import nbformat
        from nbclient import NotebookClient
        nb = nbformat.read(path, as_version=4)
        start = time.perf_counter()
        # A temporary kernelspec explicitly binds the kernel to this interpreter.
        import tempfile
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        with tempfile.TemporaryDirectory(prefix="fuzzy-kernel-") as directory:
            kernel_path = Path(directory) / "fuzzy"
            kernel_path.mkdir()
            (kernel_path / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "--matplotlib=inline", "-f", "{connection_file}"],
                "display_name": "Fuzzy notebook checks", "language": "python",
            }))
            manager = KernelManager(kernel_name="fuzzy", kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
            client = NotebookClient(nb, km=manager, timeout=args.timeout,
                                    resources={"metadata": {"path": str(path.parent)}})
            try:
                client.execute()
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
        count = sum(c.cell_type == "code" for c in nb.cells)
        images = sum("image/png" in output.get("data", {})
                     for cell in nb.cells for output in cell.get("outputs", []))
        print(f"PASS {path.name}: {count} cells, {images} figures, {time.perf_counter()-start:.2f}s")
    if args.write or args.output_dir:
        import nbformat
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, args.output_dir / path.name)
        else:
            nbformat.write(nb, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="Save execution outputs without changing source cells")
    p.add_argument("--output-dir", type=Path, help="Save executed notebooks separately, keeping repository sources unchanged")
    p.add_argument("--kernel-free", action="store_true", help="Execute Python cells and render figures without a Jupyter kernel")
    p.add_argument("--timeout", type=int, default=600, help="Per-cell Jupyter timeout in seconds")
    p.add_argument("--notebook", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.notebook:
        _worker(args)
        return
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        command = [sys.executable, str(Path(__file__).resolve()), "--notebook", str(path), "--timeout", str(args.timeout)]
        if args.kernel_free:
            command.append("--kernel-free")
        if args.write:
            command.append("--write")
        if args.output_dir:
            command.extend(["--output-dir", str(args.output_dir.resolve())])
        subprocess.run(command, cwd=path.parent, check=True)


if __name__ == "__main__":
    main()
