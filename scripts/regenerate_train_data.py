"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model's output distribution.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

# Global variables will be initialized in main function
MODEL = None
MAX_TOKENS = None
BATCH_SIZE = None
TEMPERATURE = None
BASE_URL = None
HEADERS = {"Content-Type": "application/json"}
SERVER_PROCESS = None

PARQUET_WRITE_BATCH_SIZE = 64
PARQUET_COMPRESSION = "snappy"
STATE_FILE_SUFFIX = ".state.json"
PARQUET_COLUMNS = [
    "input_line_index",
    "sample_id",
    "model_name",
    "max_tokens",
    "temperature",
    "processed_timestamp",
    "conversations_json",
]

# set env vars needed for sglang
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.6"
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + ":" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.path.join(os.environ["CUDA_HOME"], "lib64") + ":" + os.environ.get("LD_LIBRARY_PATH", "")



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens (default: 8192)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-file-path", type=str, required=True)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument(
        "--auto-launch-server",
        action="store_true",
        help="Automatically launch sglang server if port is available",
    )
    parser.add_argument("--num-samples", type=int, default=None)

    return parser.parse_args()


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True


def launch_sglang_server(
    model_path: str,
    port: int,
    mem_fraction_static: float,
) -> subprocess.Popen:
    """Launch sglang server"""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "768",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 8}',
        "--port",
        str(port),
    ]

    print("Launching sglang server with command:")
    print(" ".join(cmd))

    # Start the server process
    process = subprocess.Popen(cmd)
    return process


def wait_for_server_ready(port: int, timeout: int = 3600) -> bool:
    """Wait for server to be ready"""
    print(f"Waiting for server to be ready at localhost:{port}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_port_in_use(int(port)):
            # Port is in use, try to make a simple request
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
        time.sleep(5)

    print(f"Server failed to start within {timeout} seconds")
    return False


def cleanup_server():
    """Clean up server process"""
    global SERVER_PROCESS
    if SERVER_PROCESS and SERVER_PROCESS.poll() is None:
        print("Shutting down sglang server...")
        SERVER_PROCESS.terminate()
        try:
            SERVER_PROCESS.wait(timeout=30)
        except subprocess.TimeoutExpired:
            SERVER_PROCESS.kill()
        print("Server shutdown complete")


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal, cleaning up...")
    cleanup_server()
    sys.exit(0)


def call_sglang_batch(
    messages_list: List[List[Dict[str, Any]]]
) -> List[str]:
    """Send a batch of message lists to sglang /v1/chat/completions for vision models."""
    global MODEL, MAX_TOKENS, TEMPERATURE, BASE_URL, HEADERS

    # For vision models, we send each conversation separately
    # sglang's batch endpoint for chat/completions accepts list of message lists
    results = []
    for messages in messages_list:
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        resp = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        results.append(content.strip())
    
    return results


def get_resume_state_path(parquet_path: str) -> Path:
    """Return the path where resume state is persisted."""
    return Path(f"{parquet_path}{STATE_FILE_SUFFIX}")


def load_resume_state(state_path: Path, input_file: str) -> int:
    """Load the number of lines already processed for the provided input file."""
    if not state_path.exists():
        return 0
    try:
        with state_path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"Warning: failed to read resume state from {state_path}. "
            f"Starting from beginning. Error: {exc}"
        )
        return 0

    if state.get("input_file") != input_file:
        print(
            f"Warning: resume state {state_path} was created for "
            f"{state.get('input_file')}, not {input_file}. Ignoring it."
        )
        return 0

    processed_count = state.get("processed_line_count")
    if not isinstance(processed_count, int) or processed_count < 0:
        print(
            f"Warning: resume state {state_path} is missing a valid processed_line_count."
        )
        return 0
    return processed_count


def save_resume_state(state_path: Path, processed_count: int, input_file: str) -> None:
    """Persist the processed line count so execution can resume later."""
    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    state_payload = {
        "processed_line_count": processed_count,
        "input_file": input_file,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with temp_path.open("w", encoding="utf-8") as temp_file:
        json.dump(state_payload, temp_file)
    os.replace(temp_path, state_path)


def build_rows_for_parquet(
    batch_data: List[Dict[str, Any]],
    outputs: List[str],
    line_indices: List[int],
) -> List[Dict[str, Any]]:
    """Construct the row payload that will be written to Parquet."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    rows: List[Dict[str, Any]] = []
    for record, completion, line_id in zip(batch_data, outputs, line_indices):
        assistant_message = {"role": "assistant", "content": completion}
        record["conversations"].append(assistant_message)
        conversations_json = json.dumps(record["conversations"], ensure_ascii=False)

        rows.append(
            {
                "input_line_index": line_id,
                "sample_id": record.get("id"),
                "model_name": MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "processed_timestamp": timestamp,
                "conversations_json": conversations_json,
            }
        )
    return rows


def write_records_to_parquet(
    records: List[Dict[str, Any]],
    parquet_path: Path,
    append: bool,
) -> None:
    """Write the provided records to the Parquet sink."""
    if not records:
        return
    df = pd.DataFrame.from_records(records, columns=PARQUET_COLUMNS)
    df["input_line_index"] = df["input_line_index"].astype("int64")
    df["max_tokens"] = df["max_tokens"].astype("int64")
    df["temperature"] = df["temperature"].astype("float64")

    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
        index=False,
        append=append,
    )


def main():
    global MODEL, MAX_TOKENS, BATCH_SIZE, TEMPERATURE, BASE_URL, SERVER_PROCESS

    # Parse command line arguments
    args = parse_arguments()

    # Set global variables
    MODEL = args.model
    MAX_TOKENS = args.max_tokens
    BATCH_SIZE = args.batch_size
    TEMPERATURE = args.temperature
    BASE_URL = f"http://localhost:{args.port}/v1/chat/completions"
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    parquet_path = Path(output_file_path)
    resume_state_path = get_resume_state_path(output_file_path)

    # Validate parameters
    if not (0.0 <= TEMPERATURE <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if MAX_TOKENS <= 0:
        raise ValueError("Max tokens must be greater than 0")

    if BATCH_SIZE <= 0:
        raise ValueError("Batch size must be greater than 0")

    # Check if server needs to be launched
    if args.auto_launch_server:
        port = args.port
        if not is_port_in_use(port):
            print(f"Port {port} is available, launching sglang server...")
            try:
                SERVER_PROCESS = launch_sglang_server(
                    model_path=args.model,
                    port=port,
                    mem_fraction_static=args.mem_fraction_static,
                )

                # Wait for server to be ready
                if not wait_for_server_ready(port):
                    cleanup_server()
                    raise RuntimeError("Failed to start server")

                print("Server launched successfully!")
            except Exception as e:
                print(f"Failed to launch server: {e}")
                sys.exit(1)
        else:
            print(f"Port {port} is already in use, assuming server is running")
    else:
        port = args.port
        if not is_port_in_use(port):
            print(
                f"Warning: Port {port} is not in use. Please ensure sglang server is running."
            )

    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Configuration:")
    print(f"  Model path: {MODEL}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  API URL: {BASE_URL}")
    print(f"  Input file: {input_file_path}")
    print(f"  Output Parquet: {output_file_path}")
    print("-" * 50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print("Counting total lines in file...")
    with open(input_file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    target_total = (
        min(args.num_samples, total_lines) if args.num_samples else total_lines
    )

    resume_offset = load_resume_state(resume_state_path, input_file_path)
    if resume_offset:
        print(f"Resuming from input line offset {resume_offset}")

    if resume_offset >= target_total:
        print(
            "Nothing to process: existing resume offset is greater than or equal to "
            "requested processing limit."
        )
        cleanup_server()
        return

    remaining_to_process = target_total - resume_offset
    print(
        f"Total {remaining_to_process} lines to process in this run (of {target_total})"
    )

    pbar = tqdm(total=remaining_to_process, desc="Processing", unit="item")

    processed_total = resume_offset
    processed_this_run = 0
    parquet_append = parquet_path.exists()
    pending_rows: List[Dict[str, Any]] = []

    try:
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            batch_messages: List[List[Dict[str, Any]]] = []
            batch_data: List[Dict[str, Any]] = []
            batch_line_indices: List[int] = []

            for line_index, line in enumerate(input_file):
                if line_index < resume_offset:
                    continue
                if line_index >= target_total:
                    break

                data = json.loads(line)
                messages = data["conversations"].copy()

                # Remove last assistant message if present
                if messages and messages[-1]["role"] == "assistant":
                    messages.pop()
                
                # For Qwen3-VL: Add <image> token to first user message if image exists
                if "image" in data and data["image"] and messages:
                    for msg in messages:
                        if msg["role"] == "user":
                            # Prepend <image> token to content
                            msg["content"] = [
                                {"type": "image", "image": data["image"]},
                                {"type": "text", "text": msg["content"]}
                            ]
                            break

                batch_messages.append(messages)
                batch_data.append(data)
                batch_line_indices.append(line_index)

                if len(batch_messages) == BATCH_SIZE:
                    outputs = call_sglang_batch(batch_messages)
                    rows = build_rows_for_parquet(
                        batch_data, outputs, batch_line_indices
                    )
                    pending_rows.extend(rows)

                    while len(pending_rows) >= PARQUET_WRITE_BATCH_SIZE:
                        chunk = pending_rows[:PARQUET_WRITE_BATCH_SIZE]
                        write_records_to_parquet(chunk, parquet_path, parquet_append)
                        parquet_append = True
                        pending_rows = pending_rows[PARQUET_WRITE_BATCH_SIZE:]
                        processed_total += PARQUET_WRITE_BATCH_SIZE
                        processed_this_run += PARQUET_WRITE_BATCH_SIZE
                        pbar.update(PARQUET_WRITE_BATCH_SIZE)
                        save_resume_state(
                            resume_state_path, processed_total, input_file_path
                        )

                    batch_messages = []
                    batch_data = []
                    batch_line_indices = []

            if batch_messages:
                outputs = call_sglang_batch(batch_messages)
                rows = build_rows_for_parquet(batch_data, outputs, batch_line_indices)
                pending_rows.extend(rows)

            if pending_rows:
                write_records_to_parquet(pending_rows, parquet_path, parquet_append)
                processed_total += len(pending_rows)
                processed_this_run += len(pending_rows)
                pbar.update(len(pending_rows))
                save_resume_state(resume_state_path, processed_total, input_file_path)

        pbar.close()
        print(
            f"\nProcessing completed! Total {processed_this_run} new lines processed "
            f"(cumulative processed lines: {processed_total})"
        )

    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        if not pbar.disable:
            pbar.close()
        cleanup_server()


if __name__ == "__main__":
    main()
