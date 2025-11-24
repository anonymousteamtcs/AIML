import json
from typing import Callable, Dict, Iterator, Any, Optional

def iterate_json_rows(
    path: str,
    list_key: Optional[str] = None,
    row_handler: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Iterator[Dict[str, Any]]:
    """
    Read JSON data from file at path and yield rows as dictionaries.
    - If file is NDJSON (one JSON object per line) it parses each line.
    - If root is a list it yields each item (must be dict-like).
    - If root is an object and list_key is provided it yields items from root[list_key].
    - If row_handler is provided it's called for each yielded row.
    """
    def _yield_row(row):
        if not isinstance(row, dict):
            # convert simple values to a dict with a default key
            yield {"value": row}
        else:
            yield row

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                return  # empty file, nothing to yield

            # Try to detect NDJSON: multiple JSON objects separated by newlines
            lines = text.splitlines()
            if len(lines) > 1:
                # attempt to parse each line separately; fallback to whole-file JSON
                all_lines_parseable = True
                parsed_lines = []
                for ln in lines:
                    ln_stripped = ln.strip()
                    if not ln_stripped:
                        continue
                    try:
                        parsed_lines.append(json.loads(ln_stripped))
                    except json.JSONDecodeError:
                        all_lines_parseable = False
                        break
                if all_lines_parseable:
                    for item in parsed_lines:
                        for r in _yield_row(item):
                            if row_handler:
                                row_handler(r)
                            yield r
                    return

            # Not NDJSON, parse whole file
            data = json.loads(text)

            # If root is a list, yield each element
            if isinstance(data, list):
                for item in data:
                    for r in _yield_row(item):
                        if row_handler:
                            row_handler(r)
                        yield r
                return

            # If root is dict and a list_key was given, try to use it
            if isinstance(data, dict) and list_key and list_key in data and isinstance(data[list_key], list):
                for item in data[list_key]:
                    for r in _yield_row(item):
                        if row_handler:
                            row_handler(r)
                        yield r
                return

            # If root is a single dict, yield it as the only row
            if isinstance(data, dict):
                for r in _yield_row(data):
                    if row_handler:
                        row_handler(r)
                    yield r
                return

    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {path}: {e}") from e
