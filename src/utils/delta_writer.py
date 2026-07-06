"""
Minimal Delta Lake Writer/Reader for Development
==================================================

Produces Delta Lake-compatible table structure with:
  - _delta_log/ directory with JSON transaction log entries
  - Data files (JSON-lines format, named as .json part files)
  - Proper protocol, metadata, and add/remove actions

When PySpark + delta-spark are available, the upstream code uses them directly.
This module provides a pure-Python fallback that produces a valid Delta table
structure readable by any Delta Lake client.

Delta Lake Protocol Reference:
  - Version 0 JSON log entries
  - Protocol action: minReaderVersion=1, minWriterVersion=2
  - Metadata action: schema, partition columns, format
  - Add action: path, size, modification time, data change flag
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DeltaSchema:
    """Delta Lake table schema definition."""

    fields: list[dict[str, Any]]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "DeltaSchema":
        """Infer schema from a list of record dicts."""
        if not records:
            return cls(fields=[])

        sample = records[0]
        fields = []
        for key, value in sample.items():
            delta_type = _python_type_to_delta(value)
            fields.append({
                "name": key,
                "type": delta_type,
                "nullable": True,
                "metadata": {},
            })
        return cls(fields=fields)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Delta schema format."""
        return {
            "type": "struct",
            "fields": self.fields,
        }


def _python_type_to_delta(value: Any) -> str:
    """Map Python type to Delta Lake type string."""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "long"
    elif isinstance(value, float):
        return "double"
    elif isinstance(value, str):
        return "string"
    elif value is None:
        return "string"
    else:
        return "string"


class DeltaWriter:
    """
    Writes data as a Delta Lake table (pure-Python implementation).

    Creates:
      table_path/
        _delta_log/
          00000000000000000000.json  (commit log)
        part-00000-{uuid}.json      (data files)
        ...
    """

    def __init__(self, table_path: str | Path):
        self.table_path = Path(table_path)
        self.delta_log_path = self.table_path / "_delta_log"

    def write(
        self,
        records: list[dict[str, Any]],
        partition_by: str | None = None,
        mode: str = "overwrite",
    ) -> None:
        """
        Write records to Delta table.

        Args:
            records: List of dictionaries to write
            partition_by: Optional column to partition by
            mode: "overwrite" (default) or "append"
        """
        if mode == "overwrite" and self.table_path.exists():
            shutil.rmtree(self.table_path)

        self.table_path.mkdir(parents=True, exist_ok=True)
        self.delta_log_path.mkdir(parents=True, exist_ok=True)

        # Write data files
        if partition_by and records:
            data_files = self._write_partitioned(records, partition_by)
        else:
            data_files = self._write_unpartitioned(records)

        # Write transaction log
        schema = DeltaSchema.from_records(records)
        version = self._next_version()
        self._write_commit_log(version, schema, data_files, partition_by)

    def _write_partitioned(
        self, records: list[dict[str, Any]], partition_col: str
    ) -> list[dict[str, Any]]:
        """Write records partitioned by a column."""
        partitions: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            key = record.get(partition_col, "__null__")
            partitions.setdefault(key, []).append(record)

        data_files = []
        for idx, (partition_value, partition_records) in enumerate(
            sorted(partitions.items(), key=lambda x: str(x[0]))
        ):
            partition_dir = self.table_path / f"{partition_col}={partition_value}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"part-{idx:05d}-{_pseudo_uuid()}.json"
            file_path = partition_dir / file_name
            relative_path = f"{partition_col}={partition_value}/{file_name}"

            # Write records (without partition column in the data)
            with open(file_path, "w") as f:
                for record in partition_records:
                    row = {k: v for k, v in record.items() if k != partition_col}
                    f.write(json.dumps(row, default=str) + "\n")

            file_size = file_path.stat().st_size
            data_files.append({
                "path": relative_path,
                "size": file_size,
                "modificationTime": int(time.time() * 1000),
                "dataChange": True,
                "partitionValues": {partition_col: str(partition_value)},
                "stats": json.dumps({
                    "numRecords": len(partition_records),
                }),
            })

        return data_files

    def _write_unpartitioned(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Write all records to a single data file."""
        file_name = f"part-00000-{_pseudo_uuid()}.json"
        file_path = self.table_path / file_name

        with open(file_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

        file_size = file_path.stat().st_size
        return [{
            "path": file_name,
            "size": file_size,
            "modificationTime": int(time.time() * 1000),
            "dataChange": True,
            "partitionValues": {},
            "stats": json.dumps({"numRecords": len(records)}),
        }]

    def _write_commit_log(
        self,
        version: int,
        schema: DeltaSchema,
        data_files: list[dict[str, Any]],
        partition_by: str | None,
    ) -> None:
        """Write a Delta transaction log commit."""
        log_file = self.delta_log_path / f"{version:020d}.json"

        actions = []

        # Protocol action (first commit only)
        if version == 0:
            actions.append(json.dumps({
                "protocol": {
                    "minReaderVersion": 1,
                    "minWriterVersion": 2,
                }
            }))

            # Metadata action
            partition_columns = [partition_by] if partition_by else []
            actions.append(json.dumps({
                "metaData": {
                    "id": _pseudo_uuid(),
                    "format": {"provider": "json", "options": {}},
                    "schemaString": json.dumps(schema.to_dict()),
                    "partitionColumns": partition_columns,
                    "configuration": {},
                    "createdTime": int(time.time() * 1000),
                }
            }))

        # Add actions for each data file
        for file_info in data_files:
            actions.append(json.dumps({"add": file_info}))

        # Commit info
        actions.append(json.dumps({
            "commitInfo": {
                "timestamp": int(time.time() * 1000),
                "operation": "WRITE",
                "operationParameters": {
                    "mode": "Overwrite",
                    "partitionBy": f"[{partition_by}]" if partition_by else "[]",
                },
                "readVersion": version - 1 if version > 0 else -1,
                "isolationLevel": "Serializable",
                "isBlindAppend": False,
                "engineInfo": "cre-distress-warning-python-delta-writer-1.0",
            }
        }))

        with open(log_file, "w") as f:
            f.write("\n".join(actions) + "\n")

    def _next_version(self) -> int:
        """Get the next version number for the transaction log."""
        if not self.delta_log_path.exists():
            return 0
        existing = list(self.delta_log_path.glob("*.json"))
        if not existing:
            return 0
        versions = [int(f.stem) for f in existing]
        return max(versions) + 1


class DeltaReader:
    """
    Reads a Delta Lake table written by DeltaWriter (pure-Python).

    Reads the transaction log to determine active files, then reads data.
    """

    def __init__(self, table_path: str | Path):
        self.table_path = Path(table_path)
        self.delta_log_path = self.table_path / "_delta_log"

    def read(self) -> list[dict[str, Any]]:
        """Read all records from the Delta table."""
        if not self.delta_log_path.exists():
            raise FileNotFoundError(
                f"Not a Delta table: {self.table_path} (no _delta_log/ found)"
            )

        active_files = self._get_active_files()
        records = []

        for file_info in active_files:
            file_path = self.table_path / file_info["path"]
            partition_values = file_info.get("partitionValues", {})

            if file_path.exists():
                with open(file_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            # Add partition values back to record
                            record.update(partition_values)
                            records.append(record)

        return records

    def read_partition(self, partition_col: str, partition_value: Any) -> list[dict[str, Any]]:
        """Read records from a specific partition."""
        active_files = self._get_active_files()
        records = []

        for file_info in active_files:
            pvals = file_info.get("partitionValues", {})
            if pvals.get(partition_col) == str(partition_value):
                file_path = self.table_path / file_info["path"]
                if file_path.exists():
                    with open(file_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                record = json.loads(line)
                                record.update(pvals)
                                records.append(record)

        return records

    def count(self) -> int:
        """Get total record count from stats."""
        active_files = self._get_active_files()
        total = 0
        for file_info in active_files:
            stats = file_info.get("stats", "{}")
            if isinstance(stats, str):
                stats = json.loads(stats)
            total += stats.get("numRecords", 0)
        return total

    def get_schema(self) -> dict[str, Any] | None:
        """Get the table schema from metadata."""
        for log_file in sorted(self.delta_log_path.glob("*.json")):
            with open(log_file) as f:
                for line in f:
                    action = json.loads(line.strip())
                    if "metaData" in action:
                        schema_str = action["metaData"].get("schemaString", "{}")
                        return json.loads(schema_str)
        return None

    def _get_active_files(self) -> list[dict[str, Any]]:
        """Parse transaction log to get currently active files."""
        adds: dict[str, dict[str, Any]] = {}
        removes: set[str] = set()

        for log_file in sorted(self.delta_log_path.glob("*.json")):
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    action = json.loads(line)
                    if "add" in action:
                        path = action["add"]["path"]
                        adds[path] = action["add"]
                    elif "remove" in action:
                        path = action["remove"]["path"]
                        removes.add(path)
                        adds.pop(path, None)

        return [v for k, v in adds.items() if k not in removes]


def _pseudo_uuid() -> str:
    """Generate a pseudo-UUID using stdlib (deterministic not required here)."""
    import random
    import string

    chars = string.hexdigits[:16]
    parts = [
        "".join(random.choices(chars, k=8)),
        "".join(random.choices(chars, k=4)),
        "".join(random.choices(chars, k=4)),
        "".join(random.choices(chars, k=4)),
        "".join(random.choices(chars, k=12)),
    ]
    return "-".join(parts)
