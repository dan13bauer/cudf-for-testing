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

// Expected outcome for a matrix test case.
enum class expected_outcome {
  ROWS_3,  // filter works, 3 rows returned
  ROWS_0,  // filter passes validation but evaluates against raw values, 0 rows
  THROWS,  // scalar type != output type, throws non-matching operand types
};

char const* outcome_name(expected_outcome o)
{
  switch (o) {
    case expected_outcome::ROWS_3: return "3 rows";
    case expected_outcome::ROWS_0: return "0 rows";
    case expected_outcome::THROWS: return "throws";
  }
  return "?";
}

struct matrix_case {
  std::string col_name;         // Parquet column (determines native precision)
  cudf::type_id filter_type;    // scalar precision for the filter
  cudf::type_id timestamp_type; // timestamp_type option; EMPTY = unset
  expected_outcome expected;
};

/// Run a single matrix test case. Returns true if the actual outcome matches expected.
bool run_matrix_case(std::string const& filepath, matrix_case const& mc)
{
  // Select the scalar ticks for the filter precision.
  int64_t lower = 0, upper = 0;
  switch (mc.filter_type) {
    case cudf::type_id::TIMESTAMP_MILLISECONDS: lower = kLowerMs; upper = kUpperMs; break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: lower = kLowerUs; upper = kUpperUs; break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS: lower = kLowerNs; upper = kUpperNs; break;
    default: break;
  }

  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  cudf::ast::expression const* filter = nullptr;
  switch (mc.filter_type) {
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      filter = &build_range_filter<cudf::timestamp_ms>(0, lower, upper, tree, scalars);
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      filter = &build_range_filter<cudf::timestamp_us>(0, lower, upper, tree, scalars);
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      filter = &build_range_filter<cudf::timestamp_ns>(0, lower, upper, tree, scalars);
      break;
    default: return false;
  }

  auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                   .column_names({mc.col_name})
                   .filter(*filter);

  if (mc.timestamp_type != cudf::type_id::EMPTY) {
    builder.timestamp_type(cudf::data_type{mc.timestamp_type});
  }

  bool threw     = false;
  int64_t rows   = -1;
  try {
    auto result = cudf::io::read_parquet(builder.build());
    rows        = result.tbl->view().num_rows();
  } catch (std::exception const&) {
    threw = true;
  }

  expected_outcome actual;
  if (threw) {
    actual = expected_outcome::THROWS;
  } else if (rows == 3) {
    actual = expected_outcome::ROWS_3;
  } else if (rows == 0) {
    actual = expected_outcome::ROWS_0;
  } else {
    // unexpected row count
    std::cout << "  UNEXPECTED: " << rows << " rows\n";
    return false;
  }

  bool ok = (actual == mc.expected);
  std::cout << (ok ? "  PASS" : "  FAIL") << ": actual=" << outcome_name(actual)
            << ", expected=" << outcome_name(mc.expected) << "\n";
  return ok;
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

  int v_passed = 0;
  int v_failed = 0;
  auto check = [&](char const* label, bool condition) {
    if (condition) {
      std::cout << "  VERIFIED: " << label << "\n";
      ++v_passed;
    } else {
      std::cout << "  REFUTED:  " << label << "\n";
      ++v_failed;
    }
  };

  // =====================================================================
  // CLAIM 1: Output column type matches the requested timestamp_type,
  //          regardless of the Parquet native precision.
  // =====================================================================
  std::cout << "=== CLAIM 1: Output column type matches timestamp_type ===\n\n";
  for (auto const& [col, native_name] :
       std::vector<std::pair<std::string, std::string>>{
         {"ts_us", "us"}, {"ts_ms", "ms"}, {"ts_ns", "ns"}}) {
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({col})
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    auto result     = cudf::io::read_parquet(builder.build());
    auto output_tid = result.tbl->view().column(0).type().id();
    std::cout << "  " << col << " (native " << native_name
              << ") -> output type_id: " << type_id_name(output_tid) << "\n";
    check((col + ": output is TIMESTAMP_MILLISECONDS").c_str(),
          output_tid == cudf::type_id::TIMESTAMP_MILLISECONDS);
  }
  std::cout << "\n";

  // =====================================================================
  // CLAIM 2: AST filter literal type must match the OUTPUT type (not the
  //          native Parquet type). Mismatched literals throw.
  // =====================================================================
  std::cout << "=== CLAIM 2: Filter literal type must match output type ===\n\n";
  {
    // 2a: ms scalars + timestamp_type=ms on us column -> accepted (no throw)
    std::cout << "  2a: ms scalars on ts_us with timestamp_type=ms\n";
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    auto& filter = build_range_filter<cudf::timestamp_ms>(0, kLowerMs, kUpperMs, tree, scalars);
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({"ts_us"})
                     .filter(filter)
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    bool accepted = false;
    try {
      cudf::io::read_parquet(builder.build());
      accepted = true;
    } catch (...) {
    }
    check("ms scalars + ms output type: accepted (no throw)", accepted);
  }
  {
    // 2b: us scalars (native) + timestamp_type=ms on us column -> throws
    std::cout << "  2b: us scalars on ts_us with timestamp_type=ms\n";
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    auto& filter = build_range_filter<cudf::timestamp_us>(0, kLowerUs, kUpperUs, tree, scalars);
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({"ts_us"})
                     .filter(filter)
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    bool threw = false;
    std::string what;
    try {
      cudf::io::read_parquet(builder.build());
    } catch (std::exception const& e) {
      threw = true;
      what  = e.what();
    }
    std::cout << "      threw: " << (threw ? "yes" : "no");
    if (threw) std::cout << " (" << what << ")";
    std::cout << "\n";
    check("us scalars + ms output type: throws non-matching operand types", threw);
  }
  {
    // 2c: ns scalars (native) + timestamp_type=ms on ns column -> throws
    std::cout << "  2c: ns scalars on ts_ns with timestamp_type=ms\n";
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    auto& filter = build_range_filter<cudf::timestamp_ns>(0, kLowerNs, kUpperNs, tree, scalars);
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({"ts_ns"})
                     .filter(filter)
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    bool threw = false;
    try {
      cudf::io::read_parquet(builder.build());
    } catch (...) {
      threw = true;
    }
    check("ns scalars + ms output type: throws non-matching operand types", threw);
  }
  std::cout << "\n";

  // =====================================================================
  // CLAIM 3: Filter evaluation happens against RAW Parquet values, not
  //          converted values. ms scalars that should match 3 rows
  //          return 0 when native precision differs.
  // =====================================================================
  std::cout << "=== CLAIM 3: Filter evaluates against raw (unconverted) values ===\n\n";
  {
    // 3a: ts_us native=us, timestamp_type=ms, ms scalars -> 0 rows
    std::cout << "  3a: ts_us (native us) -> ms, filter with ms scalars\n";
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    auto& filter = build_range_filter<cudf::timestamp_ms>(0, kLowerMs, kUpperMs, tree, scalars);
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({"ts_us"})
                     .filter(filter)
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    auto result = cudf::io::read_parquet(builder.build());
    auto rows   = result.tbl->view().num_rows();
    std::cout << "      rows=" << rows << " (expected 3 if filter used converted values)\n";
    check("ts_us -> ms with ms filter: 0 rows (filter sees raw us values)", rows == 0);
  }
  {
    // 3b: ts_ns native=ns, timestamp_type=ms, ms scalars -> 0 rows
    std::cout << "  3b: ts_ns (native ns) -> ms, filter with ms scalars\n";
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    auto& filter = build_range_filter<cudf::timestamp_ms>(0, kLowerMs, kUpperMs, tree, scalars);
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({"ts_ns"})
                     .filter(filter)
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    auto result = cudf::io::read_parquet(builder.build());
    auto rows   = result.tbl->view().num_rows();
    std::cout << "      rows=" << rows << " (expected 3 if filter used converted values)\n";
    check("ts_ns -> ms with ms filter: 0 rows (filter sees raw ns values)", rows == 0);
  }
  {
    // 3c: Control -- ts_ms native=ms, timestamp_type=ms, ms scalars -> 3 rows
    //     (no conversion needed, so native and output precision match)
    std::cout << "  3c: ts_ms (native ms) -> ms, filter with ms scalars (control)\n";
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    auto& filter = build_range_filter<cudf::timestamp_ms>(0, kLowerMs, kUpperMs, tree, scalars);
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({"ts_ms"})
                     .filter(filter)
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    auto result = cudf::io::read_parquet(builder.build());
    auto rows   = result.tbl->view().num_rows();
    std::cout << "      rows=" << rows << " (native matches output, filter works)\n";
    check("ts_ms -> ms with ms filter: 3 rows (no conversion, filter correct)", rows == 3);
  }
  std::cout << "\n";

  // =====================================================================
  // CLAIM 4: Conversion happens AFTER filtering. Surviving rows appear
  //          in the requested output precision, not the native precision.
  // =====================================================================
  std::cout << "=== CLAIM 4: Conversion happens after filtering ===\n\n";
  {
    // Read ts_us (native us) with timestamp_type=ms, no filter.
    // All 5 rows survive. Output values should be in ms, not us.
    std::cout << "  4a: ts_us unfiltered with timestamp_type=ms\n";
    auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                     .column_names({"ts_us"})
                     .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    auto result     = cudf::io::read_parquet(builder.build());
    auto table      = result.tbl->view();
    auto output_tid = table.column(0).type().id();
    auto rows       = table.num_rows();

    // Read the first value and verify it's in ms range, not us range.
    std::vector<int64_t> host(rows);
    cudaMemcpy(
      host.data(), table.column(0).data<int64_t>(), rows * sizeof(int64_t), cudaMemcpyDeviceToHost);
    // First value: 2026-03-15T00:00Z = 1,773,532,800,000 ms = 1,773,532,800,000,000 us
    bool values_in_ms = (host[0] == 1'773'532'800'000LL);
    bool values_in_us = (host[0] == 1'773'532'800'000'000LL);
    std::cout << "      output type: " << type_id_name(output_tid) << "\n";
    std::cout << "      first value: " << host[0] << "\n";
    std::cout << "      is ms-scale: " << (values_in_ms ? "yes" : "no") << "\n";
    std::cout << "      is us-scale: " << (values_in_us ? "yes" : "no") << "\n";
    check("unfiltered output values are ms-scale (conversion applied)", values_in_ms);
    check("unfiltered output values are NOT us-scale", !values_in_us);
  }
  {
    // Now read with a native-matching filter (us scalars, no timestamp_type)
    // to get 3 filtered rows, then re-read with timestamp_type=ms + same
    // filter but in ms. If claim 4 is true, the ms-filtered read returns
    // 0 rows, but an unfiltered read followed by manual inspection shows
    // the conversion did happen on the full dataset.
    std::cout << "  4b: ts_us with native us filter (no conversion) -> 3 rows in us\n";
    cudf::ast::tree tree1;
    std::vector<std::unique_ptr<cudf::scalar>> scalars1;
    auto& filter1 = build_range_filter<cudf::timestamp_us>(0, kLowerUs, kUpperUs, tree1, scalars1);
    auto builder1 = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                      .column_names({"ts_us"})
                      .filter(filter1);
    auto result1 = cudf::io::read_parquet(builder1.build());
    auto rows1   = result1.tbl->view().num_rows();
    auto tid1    = result1.tbl->view().column(0).type().id();
    std::cout << "      rows=" << rows1 << ", type=" << type_id_name(tid1) << "\n";
    check("native us filter without conversion: 3 rows", rows1 == 3);
    check("native us filter output type is TIMESTAMP_MICROSECONDS",
          tid1 == cudf::type_id::TIMESTAMP_MICROSECONDS);

    std::cout << "  4c: ts_us with ms filter + timestamp_type=ms -> 0 rows (filter pre-conversion)\n";
    cudf::ast::tree tree2;
    std::vector<std::unique_ptr<cudf::scalar>> scalars2;
    auto& filter2 = build_range_filter<cudf::timestamp_ms>(0, kLowerMs, kUpperMs, tree2, scalars2);
    auto builder2 = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                      .column_names({"ts_us"})
                      .filter(filter2)
                      .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
    auto result2 = cudf::io::read_parquet(builder2.build());
    auto rows2   = result2.tbl->view().num_rows();
    auto tid2    = result2.tbl->view().column(0).type().id();
    std::cout << "      rows=" << rows2 << ", type=" << type_id_name(tid2) << "\n";
    check("ms filter + timestamp_type=ms: 0 rows (conversion is post-filter)", rows2 == 0);
    check("output type is still TIMESTAMP_MILLISECONDS despite 0 rows",
          tid2 == cudf::type_id::TIMESTAMP_MILLISECONDS);
  }
  std::cout << "\n";

  std::cout << "=== Claims verification: " << v_passed << " verified, " << v_failed
            << " refuted ===\n\n";

  // =====================================================================
  // Full 36-combination matrix:
  //   3 native precisions (us, ms, ns)
  //   x 3 filter scalar precisions (us, ms, ns)
  //   x 4 timestamp_type values (us, ms, ns, unset)
  //
  // Expected outcomes:
  //   3 rows:  filter scalar == native AND (timestamp_type == native OR unset)
  //   0 rows:  filter scalar == timestamp_type != native
  //   throws:  filter scalar != output type
  //            (output type = timestamp_type if set, else native)
  // =====================================================================

  using tid = cudf::type_id;
  using eo  = expected_outcome;

  // clang-format off
  std::vector<matrix_case> matrix = {
    // --- Parquet native: us (column ts_us) ---
    //                          col      filter                  timestamp_type          expected
    {"ts_us", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_MICROSECONDS, eo::ROWS_3},  // us/us/us
    {"ts_us", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_MILLISECONDS, eo::THROWS},  // us/us/ms
    {"ts_us", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_NANOSECONDS,  eo::THROWS},  // us/us/ns
    {"ts_us", tid::TIMESTAMP_MICROSECONDS, tid::EMPTY,                  eo::ROWS_3},  // us/us/unset
    {"ts_us", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_MICROSECONDS, eo::THROWS},  // us/ms/us
    {"ts_us", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_MILLISECONDS, eo::ROWS_0},  // us/ms/ms
    {"ts_us", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_NANOSECONDS,  eo::THROWS},  // us/ms/ns
    {"ts_us", tid::TIMESTAMP_MILLISECONDS, tid::EMPTY,                  eo::THROWS},  // us/ms/unset
    {"ts_us", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_MICROSECONDS, eo::THROWS},  // us/ns/us
    {"ts_us", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_MILLISECONDS, eo::THROWS},  // us/ns/ms
    {"ts_us", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_NANOSECONDS,  eo::ROWS_0},  // us/ns/ns
    {"ts_us", tid::TIMESTAMP_NANOSECONDS,  tid::EMPTY,                  eo::THROWS},  // us/ns/unset

    // --- Parquet native: ms (column ts_ms) ---
    {"ts_ms", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_MICROSECONDS, eo::ROWS_0},  // ms/us/us
    {"ts_ms", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_MILLISECONDS, eo::THROWS},  // ms/us/ms
    {"ts_ms", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_NANOSECONDS,  eo::THROWS},  // ms/us/ns
    {"ts_ms", tid::TIMESTAMP_MICROSECONDS, tid::EMPTY,                  eo::THROWS},  // ms/us/unset
    {"ts_ms", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_MICROSECONDS, eo::THROWS},  // ms/ms/us
    {"ts_ms", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_MILLISECONDS, eo::ROWS_3},  // ms/ms/ms
    {"ts_ms", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_NANOSECONDS,  eo::THROWS},  // ms/ms/ns
    {"ts_ms", tid::TIMESTAMP_MILLISECONDS, tid::EMPTY,                  eo::ROWS_3},  // ms/ms/unset
    {"ts_ms", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_MICROSECONDS, eo::THROWS},  // ms/ns/us
    {"ts_ms", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_MILLISECONDS, eo::THROWS},  // ms/ns/ms
    {"ts_ms", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_NANOSECONDS,  eo::ROWS_0},  // ms/ns/ns
    {"ts_ms", tid::TIMESTAMP_NANOSECONDS,  tid::EMPTY,                  eo::THROWS},  // ms/ns/unset

    // --- Parquet native: ns (column ts_ns) ---
    {"ts_ns", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_MICROSECONDS, eo::ROWS_0},  // ns/us/us
    {"ts_ns", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_MILLISECONDS, eo::THROWS},  // ns/us/ms
    {"ts_ns", tid::TIMESTAMP_MICROSECONDS, tid::TIMESTAMP_NANOSECONDS,  eo::THROWS},  // ns/us/ns
    {"ts_ns", tid::TIMESTAMP_MICROSECONDS, tid::EMPTY,                  eo::THROWS},  // ns/us/unset
    {"ts_ns", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_MICROSECONDS, eo::THROWS},  // ns/ms/us
    {"ts_ns", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_MILLISECONDS, eo::ROWS_0},  // ns/ms/ms
    {"ts_ns", tid::TIMESTAMP_MILLISECONDS, tid::TIMESTAMP_NANOSECONDS,  eo::THROWS},  // ns/ms/ns
    {"ts_ns", tid::TIMESTAMP_MILLISECONDS, tid::EMPTY,                  eo::THROWS},  // ns/ms/unset
    {"ts_ns", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_MICROSECONDS, eo::THROWS},  // ns/ns/us
    {"ts_ns", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_MILLISECONDS, eo::THROWS},  // ns/ns/ms
    {"ts_ns", tid::TIMESTAMP_NANOSECONDS,  tid::TIMESTAMP_NANOSECONDS,  eo::ROWS_3},  // ns/ns/ns
    {"ts_ns", tid::TIMESTAMP_NANOSECONDS,  tid::EMPTY,                  eo::ROWS_3},  // ns/ns/unset
  };
  // clang-format on

  int passed = 0;
  int failed = 0;

  auto ts_type_label = [](cudf::type_id t) -> std::string {
    switch (t) {
      case tid::TIMESTAMP_MICROSECONDS: return "us";
      case tid::TIMESTAMP_MILLISECONDS: return "ms";
      case tid::TIMESTAMP_NANOSECONDS: return "ns";
      case tid::EMPTY: return "unset";
      default: return "?";
    }
  };

  std::cout << "=== Full 36-combination matrix ===\n\n";
  for (int i = 0; i < static_cast<int>(matrix.size()); ++i) {
    auto const& mc = matrix[i];
    // Extract native precision from column name
    auto native = mc.col_name.substr(3);  // "us", "ms", or "ns"
    std::cout << "[" << (i + 1) << "] native=" << native
              << " filter=" << ts_type_label(mc.filter_type)
              << " timestamp_type=" << ts_type_label(mc.timestamp_type)
              << " expect=" << outcome_name(mc.expected) << "\n";
    if (run_matrix_case(filepath, mc)) {
      ++passed;
    } else {
      ++failed;
    }
  }

  std::cout << "\n=== Matrix: " << passed << " passed, " << failed << " failed, "
            << matrix.size() << " total ===\n";

  return (v_failed + failed) > 0 ? 1 : 0;
}
