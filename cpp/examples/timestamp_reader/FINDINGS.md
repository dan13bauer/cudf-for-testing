# AST Filter Pushdown and `timestamp_type` Conversion Interaction

## Summary

cuDF's Parquet reader evaluates AST filters against **raw, unconverted** column
values even when `timestamp_type` requests a precision conversion. The filter's
scalar type is validated against the *output* type (post-conversion), but the
comparison itself runs against the *raw Parquet data* (pre-conversion). This is
a consistent design choice within libcudf -- the filter operates at the storage
layer before type coercion -- but it creates a catch-22 for callers like Velox
that construct filter scalars based on the output type.

## Mechanism

When `parquet_reader_options::timestamp_type` is set (e.g., to
`TIMESTAMP_MILLISECONDS`) and the Parquet file stores timestamps at a different
native precision (e.g., microseconds), cuDF:

1. Reports the output column as the requested type (ms).
2. Validates that the AST literal type matches the output type (ms == ms, OK).
3. Evaluates the filter against the **raw Parquet values** (still in us).
4. Converts surviving rows to ms *after* filtering.

A millisecond scalar like `1,773,536,400,000` is compared against a raw
microsecond value like `1,773,536,400,000,000` -- off by 1000x -- so the range
check fails and no rows survive.

The same issue applies in both directions and to both readers:

| Parquet native | `timestamp_type` | Filter result |
|----------------|-------------------|---------------|
| us | ms | 0 rows (FAIL) |
| ns | ms | 0 rows (FAIL) |
| us | ns | 0 rows (FAIL) |
| ms | ms | 3 rows (PASS -- no conversion needed) |
| us | us | 3 rows (PASS -- no conversion needed) |

When no `timestamp_type` is set, the reader preserves native precision and
filters work correctly.

## The Catch-22

A caller cannot work around this by constructing scalars at the Parquet native
precision either. cuDF validates scalar type_ids against the *output* column
type (after `timestamp_type` conversion). Native-precision scalars (e.g., us)
on an ms output column throw `non-matching operand types`. So:

- **ms scalars + ms output type**: passes validation, but compares ms against
  raw us data -- 0 rows.
- **us scalars + ms output type**: throws `non-matching operand types`.

There is no scalar precision that produces correct results when `timestamp_type`
differs from the Parquet native precision.

## Reproducing

The `timestamp_reader` binary in this directory exercises the issue directly
against libcudf with no external dependencies:

```bash
cmake -S . -B build -Dcudf_ROOT=../../build
cmake --build build -j4
./build/timestamp_reader
```

The test Parquet file (`timestamps_multi_precision.parquet`) contains 5 rows
with identical instants stored at us, ms, and ns precision. The filter selects a
3-row window. Regenerate with `python generate_timestamp_test.py` if needed.

## Impact on Velox

Velox's `CudfHiveDataSource` sets `timestamp_type = TIMESTAMP_MILLISECONDS`
(via `CudfHiveConfig`) because Velox represents timestamps at millisecond
precision internally. It then constructs AST filter scalars at millisecond
precision to match the declared output type. cuDF accepts this without error,
but the filter silently drops all rows when the Parquet file stores timestamps
at a different native precision (the common case -- most Parquet writers default
to microseconds).

This manifests as missing data in query results: tables filtered on timestamp
ranges return empty, while non-timestamp filters (on string or integer columns)
work correctly.

## Mitigation

Because of the catch-22, AST filter pushdown cannot be used for timestamp
columns when `timestamp_type` triggers a precision conversion. The practical
mitigation in Velox is to **skip `kTimestampRange` filters from the Parquet
reader AST** and redirect them to the post-read GPU filter path
(`remainingFilter` evaluated by `cudfExpressionEvaluator_`).

Non-timestamp filters (`kBytesRange`, `kBytesValues` for string/integer columns)
still push down to the Parquet reader and handle the bulk of row elimination.
The timestamp predicate runs on the already-filtered, smaller result set --
evaluated on GPU, not on CPU. See `timestamp-filter-pushdown-fix-plan.md` in the
Velox tree for the full implementation plan.

The correct long-term fix would be in libcudf: making the filter layer aware of
the `timestamp_type` coercion so that either the filter literals or the column
data are converted to a common precision before evaluation.
