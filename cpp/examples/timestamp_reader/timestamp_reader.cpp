/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/**
 * @file timestamp_reader.cpp
 * @brief Reads a Parquet file with timestamp columns at multiple precisions,
 *        applies AST filter pushdown (column >= lower AND column <= upper),
 *        and dumps the results to stdout.
 *
 * Designed for testdata/timestamps_multi_precision.parquet:
 *   id     : INT32       (1..5)
 *   ts_us  : timestamp[us, tz=UTC]  2026-03-15 00:00..04:00
 *   ts_ms  : timestamp[ms, tz=UTC]  same instants
 *   ts_ns  : timestamp[ns, tz=UTC]  same instants
 *
 * Filter: column >= 2026-03-15T01:00 AND column <= 2026-03-15T03:00
 * Expected matching rows: 3 (hours 1, 2, 3)
 */

namespace {

// 2026-03-15T01:00:00Z
constexpr int64_t kLowerMs = 1'773'536'400'000LL;
constexpr int64_t kLowerUs = kLowerMs * 1'000LL;
constexpr int64_t kLowerNs = kLowerUs * 1'000LL;

// 2026-03-15T03:00:00Z
constexpr int64_t kUpperMs = 1'773'543'600'000LL;
constexpr int64_t kUpperUs = kUpperMs * 1'000LL;
constexpr int64_t kUpperNs = kUpperUs * 1'000LL;

constexpr int kExpectedRows = 3;

char const* type_id_name(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
    default: return "OTHER";
  }
}

void print_column_values(cudf::column_view const& col, std::string const& name)
{
  auto const num_rows = col.size();
  auto const tid      = col.type().id();

  std::cout << "    " << name << " (" << type_id_name(tid) << "): [";

  if (tid == cudf::type_id::INT32) {
    std::vector<int32_t> host(num_rows);
    cudaMemcpy(host.data(), col.data<int32_t>(), num_rows * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_rows; ++i) {
      if (i) std::cout << ", ";
      std::cout << host[i];
    }
  } else {
    std::vector<int64_t> host(num_rows);
    cudaMemcpy(host.data(), col.data<int64_t>(), num_rows * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_rows; ++i) {
      if (i) std::cout << ", ";
      std::cout << host[i];
    }
  }
  std::cout << "]\n";
}

void print_table(cudf::table_view const& table,
                 std::vector<cudf::io::column_name_info> const& schema)
{
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    print_column_values(table.column(i), schema[i].name);
  }
}

/// Build AST: column[col_idx] >= lower AND column[col_idx] <= upper.
template <typename TimestampType>
cudf::ast::expression const& build_range_filter(
  cudf::size_type col_idx,
  int64_t lower_ticks,
  int64_t upper_ticks,
  cudf::ast::tree& tree,
  std::vector<std::unique_ptr<cudf::scalar>>& scalars)
{
  using Duration = typename TimestampType::duration;

  auto lower_scalar =
    std::make_unique<cudf::timestamp_scalar<TimestampType>>(Duration(lower_ticks), true);
  auto upper_scalar =
    std::make_unique<cudf::timestamp_scalar<TimestampType>>(Duration(upper_ticks), true);

  auto& lower_lit = tree.emplace<cudf::ast::literal>(*lower_scalar);
  auto& upper_lit = tree.emplace<cudf::ast::literal>(*upper_scalar);

  scalars.push_back(std::move(lower_scalar));
  scalars.push_back(std::move(upper_scalar));

  auto& col_ref = tree.emplace<cudf::ast::column_reference>(col_idx);
  auto& ge_expr = tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::GREATER_EQUAL, col_ref, lower_lit);
  auto& le_expr =
    tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LESS_EQUAL, col_ref, upper_lit);

  return tree.emplace<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, ge_expr, le_expr);
}

struct test_case {
  std::string label;
  std::string col_name;
  cudf::type_id scalar_type;
  int64_t lower;
  int64_t upper;
  cudf::type_id timestamp_type;  // type to request via timestamp_type(); EMPTY = don't set
  int expected_rows;
  bool expect_wrong;  // true = we expect != kExpectedRows (wrong-unit test)
};

bool run_standard_read(std::string const& filepath, test_case const& tc)
{
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  cudf::ast::expression const* filter = nullptr;
  switch (tc.scalar_type) {
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      filter = &build_range_filter<cudf::timestamp_ms>(0, tc.lower, tc.upper, tree, scalars);
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      filter = &build_range_filter<cudf::timestamp_us>(0, tc.lower, tc.upper, tree, scalars);
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      filter = &build_range_filter<cudf::timestamp_ns>(0, tc.lower, tc.upper, tree, scalars);
      break;
    default: std::cerr << "  ERROR: unsupported scalar type\n"; return false;
  }

  auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                   .column_names({tc.col_name})
                   .filter(*filter);

  if (tc.timestamp_type != cudf::type_id::EMPTY) {
    builder.timestamp_type(cudf::data_type{tc.timestamp_type});
  }

  bool threw = false;
  int64_t row_count = -1;
  try {
    auto result = cudf::io::read_parquet(builder.build());
    auto table  = result.tbl->view();
    row_count   = table.num_rows();

    std::cout << "  rows=" << row_count << "\n";
    print_table(table, result.metadata.schema_info);
  } catch (std::exception const& e) {
    threw = true;
    std::cout << "  THREW: " << e.what() << "\n";
  }

  if (tc.expect_wrong) {
    if (threw) {
      std::cout << "  PASS (threw as acceptable for wrong-unit filter)\n";
      return true;
    }
    if (row_count != tc.expected_rows) {
      std::cout << "  PASS (wrong-unit filter returned " << row_count << " != " << tc.expected_rows
                << ")\n";
      return true;
    }
    std::cout << "  FAIL: wrong-unit filter unexpectedly returned correct row count\n";
    return false;
  }

  if (threw) {
    std::cout << "  FAIL: unexpected exception\n";
    return false;
  }
  if (row_count != tc.expected_rows) {
    std::cout << "  FAIL: expected " << tc.expected_rows << " rows, got " << row_count << "\n";
    return false;
  }
  std::cout << "  PASS\n";
  return true;
}

bool run_chunked_read(std::string const& filepath, test_case const& tc)
{
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  cudf::ast::expression const* filter = nullptr;
  switch (tc.scalar_type) {
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      filter = &build_range_filter<cudf::timestamp_ms>(0, tc.lower, tc.upper, tree, scalars);
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      filter = &build_range_filter<cudf::timestamp_us>(0, tc.lower, tc.upper, tree, scalars);
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      filter = &build_range_filter<cudf::timestamp_ns>(0, tc.lower, tc.upper, tree, scalars);
      break;
    default: std::cerr << "  ERROR: unsupported scalar type\n"; return false;
  }

  auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                   .column_names({tc.col_name})
                   .filter(*filter);

  if (tc.timestamp_type != cudf::type_id::EMPTY) {
    builder.timestamp_type(cudf::data_type{tc.timestamp_type});
  }

  bool threw = false;
  int64_t total_rows = 0;
  int chunks         = 0;
  try {
    constexpr std::size_t kChunkReadLimit = 0;
    cudf::io::chunked_parquet_reader reader(kChunkReadLimit, builder.build());

    while (reader.has_next()) {
      auto chunk = reader.read_chunk();
      auto table = chunk.tbl->view();
      total_rows += table.num_rows();
      ++chunks;
      if (chunks == 1) { print_table(table, chunk.metadata.schema_info); }
    }
    std::cout << "  rows=" << total_rows << " (in " << chunks << " chunks)\n";
  } catch (std::exception const& e) {
    threw = true;
    std::cout << "  THREW: " << e.what() << "\n";
  }

  if (threw) {
    std::cout << "  FAIL: unexpected exception\n";
    return false;
  }
  if (total_rows != tc.expected_rows) {
    std::cout << "  FAIL: expected " << tc.expected_rows << " rows, got " << total_rows << "\n";
    return false;
  }
  std::cout << "  PASS\n";
  return true;
}

/// Read without any filter, just with timestamp_type set, to verify conversion works.
void run_unfiltered_read(std::string const& filepath,
                         std::string const& col_name,
                         cudf::type_id ts_type)
{
  auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                   .column_names({col_name});

  if (ts_type != cudf::type_id::EMPTY) {
    builder.timestamp_type(cudf::data_type{ts_type});
  }

  auto result = cudf::io::read_parquet(builder.build());
  auto table  = result.tbl->view();

  std::cout << "  " << col_name << " -> " << type_id_name(ts_type) << ": " << table.num_rows()
            << " rows\n";
  print_table(table, result.metadata.schema_info);
}

}  // namespace

int main(int argc, char const** argv)
{
  // Default: look for the parquet file next to the source tree (not the build dir).
  std::string filepath = TIMESTAMP_READER_DATA_DIR "/timestamps_multi_precision.parquet";
  if (argc >= 2) { filepath = argv[1]; }

  // Pool memory resource
  auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
  auto pool_mr = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    cuda_mr, rmm::percent_of_free_device_memory(50));
  cudf::set_current_device_resource(pool_mr.get());

  // --- Baseline: unfiltered reads with timestamp_type=ms ---
  std::cout << "=== Baseline: unfiltered read with timestamp_type=ms ===\n\n";
  for (auto const& col : {"ts_us", "ts_ms", "ts_ns"}) {
    run_unfiltered_read(filepath, col, cudf::type_id::TIMESTAMP_MILLISECONDS);
  }
  std::cout << "\n";

  // --- Test cases ---

  // ConvertedToMs: read each timestamp column as ms, filter with ms scalars.
  std::vector<test_case> standard_cases = {
    // -- ConvertedToMs --
    {"ts_us -> ms (standard)", "ts_us", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs, kUpperMs,
     cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},
    {"ts_ms -> ms (standard)", "ts_ms", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs, kUpperMs,
     cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},
    {"ts_ns -> ms (standard)", "ts_ns", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs, kUpperMs,
     cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},

    // -- NativeResolution --
    {"ts_us native us (standard)", "ts_us", cudf::type_id::TIMESTAMP_MICROSECONDS, kLowerUs,
     kUpperUs, cudf::type_id::TIMESTAMP_MICROSECONDS, kExpectedRows, false},
    {"ts_ms native ms (standard)", "ts_ms", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs,
     kUpperMs, cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},
    {"ts_ns native ns (standard)", "ts_ns", cudf::type_id::TIMESTAMP_NANOSECONDS, kLowerNs,
     kUpperNs, cudf::type_id::TIMESTAMP_NANOSECONDS, kExpectedRows, false},

    // -- CrossConvert --
    {"ts_us -> us (standard)", "ts_us", cudf::type_id::TIMESTAMP_MICROSECONDS, kLowerUs, kUpperUs,
     cudf::type_id::TIMESTAMP_MICROSECONDS, kExpectedRows, false},
    {"ts_us -> ns (standard)", "ts_us", cudf::type_id::TIMESTAMP_NANOSECONDS, kLowerNs, kUpperNs,
     cudf::type_id::TIMESTAMP_NANOSECONDS, kExpectedRows, false},

    // -- WrongUnit: us scalars against ms-converted column --
    {"ts_us -> ms, WRONG us filter (standard)", "ts_us", cudf::type_id::TIMESTAMP_MICROSECONDS,
     kLowerUs, kUpperUs, cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, true},

    // -- NoConversion: no timestamp_type set, native precision --
    {"ts_us no conversion (standard)", "ts_us", cudf::type_id::TIMESTAMP_MICROSECONDS, kLowerUs,
     kUpperUs, cudf::type_id::EMPTY, kExpectedRows, false},
    {"ts_ms no conversion (standard)", "ts_ms", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs,
     kUpperMs, cudf::type_id::EMPTY, kExpectedRows, false},
    {"ts_ns no conversion (standard)", "ts_ns", cudf::type_id::TIMESTAMP_NANOSECONDS, kLowerNs,
     kUpperNs, cudf::type_id::EMPTY, kExpectedRows, false},
  };

  std::vector<test_case> chunked_cases = {
    // -- ConvertedToMs (chunked) --
    {"ts_us -> ms (chunked)", "ts_us", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs, kUpperMs,
     cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},
    {"ts_ms -> ms (chunked)", "ts_ms", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs, kUpperMs,
     cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},
    {"ts_ns -> ms (chunked)", "ts_ns", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs, kUpperMs,
     cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},

    // -- NativeResolution (chunked) --
    {"ts_us native us (chunked)", "ts_us", cudf::type_id::TIMESTAMP_MICROSECONDS, kLowerUs,
     kUpperUs, cudf::type_id::TIMESTAMP_MICROSECONDS, kExpectedRows, false},
    {"ts_ms native ms (chunked)", "ts_ms", cudf::type_id::TIMESTAMP_MILLISECONDS, kLowerMs,
     kUpperMs, cudf::type_id::TIMESTAMP_MILLISECONDS, kExpectedRows, false},
    {"ts_ns native ns (chunked)", "ts_ns", cudf::type_id::TIMESTAMP_NANOSECONDS, kLowerNs,
     kUpperNs, cudf::type_id::TIMESTAMP_NANOSECONDS, kExpectedRows, false},
  };

  int passed  = 0;
  int failed  = 0;
  int total   = 0;

  std::cout << "=== Standard reader tests ===\n\n";
  for (auto const& tc : standard_cases) {
    ++total;
    std::cout << "[" << total << "] " << tc.label << "\n";
    if (run_standard_read(filepath, tc)) {
      ++passed;
    } else {
      ++failed;
    }
    std::cout << "\n";
  }

  std::cout << "=== Chunked reader tests ===\n\n";
  for (auto const& tc : chunked_cases) {
    ++total;
    std::cout << "[" << total << "] " << tc.label << "\n";
    if (run_chunked_read(filepath, tc)) {
      ++passed;
    } else {
      ++failed;
    }
    std::cout << "\n";
  }

  std::cout << "=== Summary: " << passed << " passed, " << failed << " failed, " << total
            << " total ===\n";

  return failed > 0 ? 1 : 0;
}
