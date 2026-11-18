import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys


LATENCY_PATTERN = re.compile(r"\[\s*INFO\s*\]\s+Average:\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|us)\b", re.IGNORECASE)
THROUGHPUT_PATTERN = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*FPS", re.IGNORECASE)


def resolve_benchmark_app():
    executable = shutil.which("benchmark_app")
    if executable:
        return executable

    executable_name = "benchmark_app.exe" if sys.platform == "win32" else "benchmark_app"
    environment_executable = Path(sys.executable).resolve().parent / executable_name
    if environment_executable.is_file():
        return str(environment_executable)

    return None


def discover_model_paths(input_paths):
    model_paths = []
    for input_path in input_paths:
        if input_path.is_file():
            if input_path.suffix.lower() != ".xml":
                raise ValueError(f"model file must have an .xml extension: {input_path}")
            model_paths.append(input_path)
        elif input_path.is_dir():
            model_paths.extend(
                sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".xml")
            )
        else:
            raise ValueError(f"input path does not exist: {input_path}")

    unique_paths = {}
    for model_path in model_paths:
        unique_paths.setdefault(model_path.resolve(), model_path)
    return list(unique_paths.values())


def format_table(rows, show_throughput):
    headers = ["Model XML", "Device", "Average Latency"]
    if show_throughput:
        headers.append("Throughput (FPS)")
    headers.append("Status")
    table_rows = [headers]
    for row in rows:
        table_row = [
            str(row["model"]),
            row["device"],
            row["latency"] or "N/A",
        ]
        if show_throughput:
            table_row.append(row["throughput"] or "N/A")
        table_row.append(row["status"])
        table_rows.append(table_row)
    widths = [max(len(str(row[column])) for row in table_rows) for column in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [separator]
    for row_index, row in enumerate(table_rows):
        lines.append("| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |")
        if row_index == 0:
            lines.append(separator)
    lines.append(separator)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenVINO IR models on CPU, GPU, and NPU with benchmark_app."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="OpenVINO XML files and/or folders to search recursively")
    parser.add_argument(
        "--devices",
        nargs="+",
        type=str.upper,
        default=("CPU", "GPU", "NPU"),
        help="Devices to test (default: CPU GPU NPU)",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.txt"), help="Text report path")
    parser.add_argument("--hint", choices=("latency", "throughput", "none"), default="latency", help="OpenVINO performance hint")
    parser.add_argument("--shape", help='Override model input shape, e.g. "[1,3,1024,1024]"')
    parser.add_argument("--data-shape", help="Input data shape for models with dynamic inputs")
    parser.add_argument("--layout", help='Input layout, e.g. "[NCHW]"')
    args = parser.parse_args()

    try:
        model_paths = discover_model_paths(args.paths)
    except ValueError as error:
        parser.error(str(error))
    if not model_paths:
        parser.error("no .xml model files found in the supplied paths")

    benchmark_app = resolve_benchmark_app()
    if benchmark_app is None:
        parser.error("benchmark_app was not found; add it to PATH or the active Python environment")

    rows = []
    raw_outputs = []
    for model_path in model_paths:
        for device in args.devices:
            command = [
                benchmark_app,
                "-m",
                str(model_path),
                "-d",
                device,
                "-hint",
                args.hint,
            ]
            if args.shape:
                command.extend(("-shape", args.shape))
            if args.data_shape:
                command.extend(("-data_shape", args.data_shape))
            if args.layout:
                command.extend(("-layout", args.layout))

            try:
                completed = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
                output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
                status = "PASS" if completed.returncode == 0 else f"FAIL ({completed.returncode})"
                latency_match = LATENCY_PATTERN.search(output)
                throughput_match = THROUGHPUT_PATTERN.search(output)
                rows.append(
                    {
                        "model": model_path,
                        "device": device,
                        "latency": f"{latency_match.group(1)} {latency_match.group(2)}" if latency_match else None,
                        "throughput": throughput_match.group(1) if throughput_match else None,
                        "status": status,
                    }
                )
            except OSError as error:
                output = str(error)
                rows.append(
                    {
                        "model": model_path,
                        "device": device,
                        "latency": None,
                        "throughput": None,
                        "status": "FAILED TO START",
                    }
                )

            raw_outputs.append(
                "\n".join(
                    (
                        f"Model: {model_path}",
                        f"Device: {device}",
                        f"Command: {' '.join(command)}",
                        f"Status: {rows[-1]['status']}",
                        "Output:",
                        output or "(no output)",
                    )
                )
            )

    summary = format_table(rows, show_throughput=args.hint != "latency")
    print(summary)
    print(f"\nFull benchmark output: {args.output}")

    report = "\n".join(
        (
            "OpenVINO benchmark_app results",
            f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"benchmark_app: {benchmark_app}",
            f"Performance hint: {args.hint}",
            "",
            "Summary:",
            summary,
            "",
            "Raw benchmark outputs:",
            "\n\n".join(raw_outputs),
            "",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
