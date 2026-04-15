#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate a Parquet file with timestamp columns at different precisions.

Produces timestamps_multi_precision.parquet with:
  - id        INT32      : 1..5
  - ts_us     timestamp[us] : 5 timestamps from 2026-03-15T00:00 to T04:00
  - ts_ms     timestamp[ms] : same logical instants
  - ts_ns     timestamp[ns] : same logical instants

All three timestamp columns encode identical physical instants at different
Parquet-native precisions. This lets the C++ test verify whether cuDF's AST
filter pushdown operates on converted or raw timestamp values.
"""

import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone

timestamps = [
    datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 3, 15, 1, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 3, 15, 2, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 3, 15, 3, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 3, 15, 4, 0, 0, tzinfo=timezone.utc),
]

ids = list(range(1, 6))

table = pa.table(
    {
        "id": pa.array(ids, type=pa.int32()),
        "ts_us": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
        "ts_ms": pa.array(timestamps, type=pa.timestamp("ms", tz="UTC")),
        "ts_ns": pa.array(timestamps, type=pa.timestamp("ns", tz="UTC")),
    }
)

output_path = "timestamps_multi_precision.parquet"
pq.write_table(table, output_path, use_dictionary=False, write_statistics=True)

# Verify the written file.
meta = pq.read_metadata(output_path)
print(f"Wrote {output_path}: {meta.num_rows} rows, {meta.num_columns} columns")
for i in range(meta.num_columns):
    col = meta.row_group(0).column(i)
    print(f"  {col.path_in_schema}: physical_type={col.physical_type}")

schema = pq.read_schema(output_path)
for field in schema:
    print(f"  Arrow schema: {field.name} -> {field.type}")
