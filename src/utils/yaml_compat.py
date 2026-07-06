"""
Minimal YAML compatibility layer.

Tries to use PyYAML if available; otherwise provides a basic pure-Python
YAML parser that handles the subset of YAML used in this project's config files:
- Scalar values (strings, numbers, booleans)
- Nested dictionaries (indentation-based)
- Lists (- prefix)
- Quoted strings
- Comments (# prefix)

This is NOT a full YAML parser. It handles config/loan_generator.yaml,
config/stress_scenarios.yaml, and config/cap_rates_historical.yaml specifically.
"""

from __future__ import annotations

from typing import Any

try:
    import yaml  # type: ignore[import-untyped]

    def safe_load(text: str) -> Any:
        """Load YAML using PyYAML."""
        return yaml.safe_load(text)

    HAS_PYYAML = True

except ImportError:
    HAS_PYYAML = False

    def safe_load(text: str) -> Any:
        """Pure-Python minimal YAML parser for simple config files."""
        return _parse_yaml(text)


def load_yaml_file(path: str | Any) -> Any:
    """Load a YAML file, using PyYAML or fallback parser."""
    with open(path) as f:
        content = f.read()
    return safe_load(content)


# ─── Pure-Python YAML Parser ──────────────────────────────────────────────────


def _parse_yaml(text: str) -> Any:
    """Parse a simple YAML document into Python objects."""
    lines = text.split("\n")
    result, _ = _parse_block(lines, 0, 0)
    return result


def _parse_block(lines: list[str], start: int, base_indent: int) -> tuple[Any, int]:
    """Parse a YAML block at a given indentation level."""
    result: dict[str, Any] = {}
    i = start

    while i < len(lines):
        line = lines[i]

        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Calculate indentation
        indent = len(line) - len(line.lstrip())

        # If indentation decreased, we've left this block
        if indent < base_indent:
            break

        # If indentation is greater than expected, skip (handled by parent)
        if indent > base_indent and i > start:
            break

        # Parse key: value or key: (block follows)
        if ":" in stripped and not stripped.startswith("-"):
            colon_pos = stripped.index(":")
            key = stripped[:colon_pos].strip().strip('"').strip("'")
            value_str = stripped[colon_pos + 1:].strip()

            # Remove inline comments
            if " #" in value_str:
                value_str = value_str[:value_str.index(" #")].strip()

            if value_str == "" or value_str == "|" or value_str == ">":
                # Block value — look at next lines
                i += 1
                if i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    next_indent = len(next_line) - len(next_line.lstrip()) if next_stripped else indent + 2

                    if next_stripped and next_stripped.startswith("-"):
                        # It's a list
                        child, i = _parse_list(lines, i, next_indent)
                        result[key] = child
                    elif next_indent > indent:
                        # It's a nested dict
                        child, i = _parse_block(lines, i, next_indent)
                        result[key] = child
                    else:
                        result[key] = None
                else:
                    result[key] = None
            elif value_str.startswith("[") and value_str.endswith("]"):
                # Inline list
                inner = value_str[1:-1]
                items = [_parse_scalar(item.strip()) for item in inner.split(",") if item.strip()]
                result[key] = items
                i += 1
            elif value_str.startswith("{") and value_str.endswith("}"):
                # Inline dict
                inner = value_str[1:-1]
                d = {}
                for pair in inner.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        d[k.strip().strip('"').strip("'")] = _parse_scalar(v.strip())
                result[key] = d
                i += 1
            else:
                result[key] = _parse_scalar(value_str)
                i += 1

        elif stripped.startswith("-"):
            # We're in a list context at this level
            child, i = _parse_list(lines, i, base_indent)
            return child, i
        else:
            i += 1

    return result, i


def _parse_list(lines: list[str], start: int, base_indent: int) -> tuple[list[Any], int]:
    """Parse a YAML list."""
    result: list[Any] = []
    i = start

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = len(line) - len(line.lstrip())

        if indent < base_indent:
            break

        if stripped.startswith("- "):
            item_value = stripped[2:].strip()
            if ":" in item_value and not item_value.startswith('"'):
                # It's a dict item starting on this line
                # Parse as inline key: value, then check for more keys below
                item_dict: dict[str, Any] = {}
                colon_pos = item_value.index(":")
                key = item_value[:colon_pos].strip().strip('"').strip("'")
                val = item_value[colon_pos + 1:].strip()
                if " #" in val:
                    val = val[:val.index(" #")].strip()
                item_dict[key] = _parse_scalar(val)
                i += 1

                # Check for continuation keys at deeper indent
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    if not next_stripped or next_stripped.startswith("#"):
                        i += 1
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent > indent and ":" in next_stripped and not next_stripped.startswith("-"):
                        c_pos = next_stripped.index(":")
                        k = next_stripped[:c_pos].strip().strip('"').strip("'")
                        v = next_stripped[c_pos + 1:].strip()
                        if " #" in v:
                            v = v[:v.index(" #")].strip()
                        item_dict[k] = _parse_scalar(v)
                        i += 1
                    else:
                        break

                result.append(item_dict)
            else:
                result.append(_parse_scalar(item_value))
                i += 1
        elif stripped.startswith("-"):
            # Bare dash with nothing after (shouldn't happen in our configs)
            i += 1
        else:
            break

    return result, i


def _parse_scalar(value: str) -> Any:
    """Parse a YAML scalar value into a Python type."""
    if not value:
        return None

    # Remove inline comments
    if " #" in value:
        value = value[:value.index(" #")].strip()

    # Quoted strings
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    # Boolean
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    # None
    if value.lower() in ("null", "~", "none"):
        return None

    # Numbers
    # Handle underscores in numbers (e.g., 1_000_000)
    clean_value = value.replace("_", "")
    try:
        return int(clean_value)
    except ValueError:
        pass
    try:
        return float(clean_value)
    except ValueError:
        pass

    # Plain string
    return value
