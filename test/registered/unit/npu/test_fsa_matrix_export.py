"""Workbook exporter mapping, status labels, gains and one-time backup behaviour."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
SOURCE = ROOT / "scripts/npu/export_fsa_matrix_xlsx.py"
sys.path.insert(0, str(ROOT / "scripts/npu"))

SPEC = importlib.util.spec_from_file_location("export_fsa_matrix_xlsx", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

openpyxl = MODULE.openpyxl
styles = openpyxl.styles

HEADER_ROW_1 = {"A1": "方案", "B1": "场景", "E1": "性能"}
HEADER_ROW_2 = ["上下文", "输入/k", "输出/k", "部署方案（混部/pd分离）", "并行策略", "权重版本",
                "TTFT", "ITL", "TPOT", "输出吞吐", "单卡输出吞吐", "E2E吞吐", "单卡E2E吞吐"]
MERGES = {"A1:A2", "B1:D1", "E1:N1"}
MEASURED = [100.0, 20.0, 25.0, 50.0, 6.25, 400.5, 50.0625]
FIELD_NAMES = ("ttft_ms", "itl_ms", "tpot_ms", "output_tokens_per_s",
               "output_tokens_per_s_per_card", "e2e_tokens_per_s",
               "e2e_tokens_per_s_per_card")
BASE_METRICS = dict(zip(FIELD_NAMES, MEASURED))
SATURATION = MODULE.MODE_LABELS["saturation"]
COLUMN_COUNT = 14
FIRST_DATA_ROW = MODULE.FIRST_DATA_ROW


def metrics(**overrides):
    values = dict(BASE_METRICS)
    values.update(overrides)
    return values


def ok_case(mode, length, **overrides):
    case = {"status": "ok", "mode": mode, "input_tokens": length, "output_tokens": 128,
            "submitted_requests": 64 if mode == "saturation" else 1,
            "client_concurrency": 64 if mode == "saturation" else 1,
            "actual_max_decode_batch": 61, "actual_max_full_tokens": 1020000,
            "estimated_resident_requests": 64, "errors": [], "metrics": metrics()}
    case.update(overrides)
    return case


def no_data_case(mode, length, reason="Resolved token pool cannot fit one request"):
    return {"status": "no_data", "mode": mode, "input_tokens": length, "output_tokens": 128,
            "submitted_requests": 64, "client_concurrency": 64,
            "actual_max_decode_batch": 0, "actual_max_full_tokens": 0,
            "errors": [{"request": 7, "error": "RuntimeError('timeout')"}], "reason": reason}


def make_template(path):
    """A minimal stand-in for the provided workbook: same headers, merges and styles."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = MODULE.MATRIX_SHEET
    for coordinate, value in HEADER_ROW_1.items():
        sheet[coordinate] = value
    for index, value in enumerate(HEADER_ROW_2, start=2):
        sheet.cell(row=2, column=index, value=value)
    sheet["A3"] = "基线"
    sheet["A4"] = "offload"
    sheet["A3"].font = styles.Font(name="等线", size=13, bold=True)
    sheet["A3"].alignment = styles.Alignment(horizontal="center")
    for merged in MERGES:
        sheet.merge_cells(merged)
    workbook.create_sheet("其他数据")["A1"] = "保留"
    workbook.save(path)
    return path


def write_arm(directory, arm, cases, host="10.0.0.1"):
    directory.mkdir(parents=True, exist_ok=True)
    info = {"context_length": 1048576, "max_running_requests": 64, "page_size": 64,
            "max_total_num_tokens": 262144, "max_req_input_len": 1048576, "tp_size": 8,
            "disable_radix_cache": True, "speculative_algorithm": None,
            "model_path": "/models/glm"}
    (directory / "server_info.json").write_text(json.dumps(info), encoding="utf-8")
    final = {"arm": arm, "host": host, "status": "completed", "cases": list(cases)}
    (directory / "final.json").write_text(json.dumps(final, ensure_ascii=False), encoding="utf-8")
    for case in cases:
        case_directory = directory / f"{case['mode']}_{case['input_tokens']}"
        case_directory.mkdir(exist_ok=True)
        (case_directory / "summary.json").write_text(json.dumps(case, ensure_ascii=False),
                                                     encoding="utf-8")
    return directory


@pytest.fixture
def experiment(tmp_path):
    def build(baseline_cases, offload_cases):
        workbook = make_template(tmp_path / "offload对比.xlsx")
        baseline = write_arm(tmp_path / "baseline", "baseline", baseline_cases, host="10.0.0.1")
        offload = write_arm(tmp_path / "offload", "offload", offload_cases, host="10.0.0.2")
        return workbook, baseline, offload

    return build


def load(workbook, title):
    return openpyxl.load_workbook(workbook)[title]


def matrix_rows(workbook, title):
    sheet = load(workbook, title)
    rows = {}
    for row in sheet.iter_rows(min_row=FIRST_DATA_ROW, max_col=COLUMN_COUNT):
        if row[0].value in MODULE.ARM_LABELS.values() and isinstance(row[2].value, (int, float)):
            rows[(row[0].value, round(row[2].value, 6))] = [cell.value for cell in row]
    return rows


def sheet_values(workbook, title):
    values = []
    for row in load(workbook, title).iter_rows():
        for cell in row:
            if cell.value is not None:
                values.append(cell.value)
    return values


def detail_rows(workbook):
    sheet = load(workbook, MODULE.DETAIL_SHEET)
    rows = []
    for row in sheet.iter_rows(min_row=2, max_col=len(MODULE.DETAIL_HEADER)):
        values = [cell.value for cell in row]
        if values[0] in MODULE.MODE_LABELS.values():
            rows.append(values)
    return rows


def detail_cell(workbook, input_k, arm_label, column, mode_label=SATURATION):
    for row in detail_rows(workbook):
        if row[0] == mode_label and row[2] == input_k and row[3] == arm_label:
            return row[column]
    raise AssertionError(f"missing detail row for {mode_label} {input_k} {arm_label}")


def gain_rows(workbook):
    sheet = load(workbook, MODULE.GAIN_SHEET)
    rows = []
    for row in sheet.iter_rows(min_row=2, max_col=len(MODULE.GAIN_HEADER)):
        values = [cell.value for cell in row]
        if values[0] in MODULE.MODE_LABELS.values():
            rows.append(values)
    return rows


def test_saturation_rows_map_measured_metrics(experiment):
    workbook, baseline, offload = experiment(
        [ok_case("saturation", 128000)],
        [ok_case("saturation", 128000,
                 metrics=metrics(ttft_ms=90.0, itl_ms=18.0, tpot_ms=22.0,
                                 output_tokens_per_s=60.0, output_tokens_per_s_per_card=7.5,
                                 e2e_tokens_per_s=420.0, e2e_tokens_per_s_per_card=52.5))])
    MODULE.export_workbook(workbook, baseline, offload)
    rows = matrix_rows(workbook, MODULE.MATRIX_SHEET)
    baseline_row = rows[("基线", 128.0)]
    assert baseline_row[1] == MODULE.CONTEXT_LENGTH == 1048576
    assert baseline_row[2] == 128.0
    assert baseline_row[3] == pytest.approx(0.128)
    assert baseline_row[4] == MODULE.DEPLOYMENT
    assert baseline_row[5] == "TP8"
    assert "FP8" in baseline_row[6]
    assert baseline_row[7:] == MEASURED
    assert rows[("offload", 128.0)][7:] == [90.0, 18.0, 22.0, 60.0, 7.5, 420.0, 52.5]
    assert detail_cell(workbook, 128.0, "基线", 5) == 64
    assert detail_cell(workbook, 128.0, "基线", 7) == 61
    evidence = detail_cell(workbook, 128.0, "基线", 15)
    assert "saturation_128000" in evidence


def test_single_sheet_mirrors_template_layout(experiment):
    workbook, baseline, offload = experiment([ok_case("single", 512000)],
                                             [ok_case("single", 512000)])
    MODULE.export_workbook(workbook, baseline, offload)
    single = load(workbook, MODULE.SINGLE_SHEET)
    template = load(workbook, MODULE.MATRIX_SHEET)
    for coordinate, value in HEADER_ROW_1.items():
        assert single[coordinate].value == value == template[coordinate].value
    assert [single.cell(row=2, column=column).value for column in range(2, 15)] == HEADER_ROW_2
    assert {str(item) for item in single.merged_cells.ranges} == MERGES
    assert {str(item) for item in template.merged_cells.ranges} == MERGES
    assert matrix_rows(workbook, MODULE.SINGLE_SHEET)[("基线", 512.0)][7:] == MEASURED
    pending = matrix_rows(workbook, MODULE.MATRIX_SHEET)[("基线", 512.0)][7:]
    assert pending == [MODULE.PENDING] * 7


def test_missing_cases_are_pending_not_failed(experiment):
    workbook, baseline, offload = experiment([], [ok_case("saturation", 128000)])
    MODULE.export_workbook(workbook, baseline, offload)
    rows = matrix_rows(workbook, MODULE.MATRIX_SHEET)
    assert set(rows) == {(label, length / MODULE.DECIMAL_K)
                         for label in MODULE.ARM_LABELS.values()
                         for length in MODULE.DEFAULT_LENGTHS}
    assert rows[("基线", 128.0)][7:] == [MODULE.PENDING] * 7
    assert rows[("offload", 128.0)][7:] == MEASURED
    assert detail_cell(workbook, 128.0, "基线", 4) == MODULE.PENDING
    assert detail_cell(workbook, 128.0, "offload", 4) == "成功"
    assert all(row[4] == MODULE.PENDING for row in detail_rows(workbook) if row[3] == "基线")


def test_explicit_no_data_is_not_pending(experiment):
    failing = no_data_case("saturation", 256000)
    workbook, baseline, offload = experiment([ok_case("saturation", 128000), failing],
                                             [ok_case("saturation", 128000)])
    MODULE.export_workbook(workbook, baseline, offload)
    row = matrix_rows(workbook, MODULE.MATRIX_SHEET)[("基线", 256.0)]
    assert row[7:] == [MODULE.NO_DATA] * 7
    assert all(isinstance(value, str) for value in row[7:])
    assert detail_cell(workbook, 256.0, "基线", 4) == MODULE.NO_DATA
    note = detail_cell(workbook, 256.0, "基线", 14)
    assert "no_data" in note and "cannot fit" in note and "RuntimeError" in note
    assert detail_cell(workbook, 128.0, "基线", 4) == "成功"


def test_gains_are_only_computed_for_successful_pairs(experiment):
    baseline_cases = [ok_case("saturation", 128000), ok_case("saturation", 1000000)]
    offload_cases = [ok_case("saturation", 128000,
                             metrics=metrics(ttft_ms=80.0, output_tokens_per_s=62.5)),
                     no_data_case("saturation", 1000000)]
    workbook, baseline, offload = experiment(baseline_cases, offload_cases)
    MODULE.export_workbook(workbook, baseline, offload)
    rows = gain_rows(workbook)
    ttft = [row for row in rows if row[3] == "TTFT"]
    assert len(ttft) == 1
    assert ttft[0][:3] == [SATURATION, 1048576, 128.0]
    assert ttft[0][5] == 100.0 and ttft[0][6] == 80.0
    assert ttft[0][7] == pytest.approx(20.0)
    throughput = [row for row in rows if row[3] == "输出吞吐"]
    assert len(throughput) == 1
    assert throughput[0][7] == pytest.approx(25.0)
    assert len(rows) == len(MODULE.METRIC_COLUMNS)
    assert 1000.0 not in [row[2] for row in rows]
    notes = [value for value in sheet_values(workbook, MODULE.GAIN_SHEET)
             if isinstance(value, str)]
    assert any("1000K" in value and MODULE.NO_DATA in value for value in notes)


def test_backup_is_created_once_and_preserved(experiment):
    workbook, baseline, offload = experiment([ok_case("saturation", 128000)],
                                             [ok_case("saturation", 128000)])
    original = workbook.read_bytes()
    first = MODULE.export_workbook(workbook, baseline, offload, timestamp="20260101_010203")
    backup = Path(first["backup"])
    assert first["backup_created"] is True
    assert backup.name == "offload对比.backup-20260101_010203.xlsx"
    assert backup.read_bytes() == original
    second = MODULE.export_workbook(workbook, baseline, offload, timestamp="20260102_010203")
    assert second["backup"] == first["backup"]
    assert second["backup_created"] is False
    assert sorted(workbook.parent.glob("offload对比.backup-*.xlsx")) == [backup]
    assert backup.read_bytes() == original
    sheet = load(workbook, MODULE.MATRIX_SHEET)
    assert [sheet.cell(row=2, column=column).value for column in range(2, 15)] == HEADER_ROW_2
    assert sheet["A3"].font.bold is True and sheet["A3"].font.size == 13
    assert load(workbook, "其他数据")["A1"].value == "保留"
    assert len(matrix_rows(workbook, MODULE.MATRIX_SHEET)) == 2 * len(MODULE.DEFAULT_LENGTHS)


def test_config_sheet_documents_contract_and_hosts(experiment):
    workbook, baseline, offload = experiment([ok_case("saturation", 128000)],
                                             [ok_case("saturation", 128000)])
    MODULE.export_workbook(workbook, baseline, offload)
    text = "\n".join(str(value) for value in sheet_values(workbook, MODULE.CONFIG_SHEET))
    assert "10.0.0.1" in text and "10.0.0.2" in text
    assert str(baseline) in text and str(offload) in text
    assert "1K = 1000 tokens" in text
    assert "0.128K" in text
    assert "1048576" in text
    assert "TP8" in text and "MTP0" in text and "BF16 KV" in text
    assert "batch saturation" in text
    assert "max_running_requests = 64" in text
    assert "token/s" in text
    assert "未启用 profiling" in text
    assert "actual_max_decode_batch" in text


def test_cli_entrypoint_exports_workbook(experiment):
    workbook, baseline, offload = experiment([ok_case("saturation", 128000)], [])
    status = MODULE.main(["--workbook", str(workbook), "--baseline", str(baseline),
                          "--offload", str(offload)])
    assert status == 0
    book = openpyxl.load_workbook(workbook)
    assert set(MODULE.GENERATED_SHEETS) <= set(book.sheetnames)


def test_locked_workbook_uses_results_copy_and_preserves_original(experiment, monkeypatch):
    workbook, baseline, offload = experiment([ok_case("single", 128000)], [])
    original = workbook.read_bytes()
    replace = Path.replace

    def locked_replace(source, target):
        if Path(target) == workbook:
            raise PermissionError("Workbook is open in another application")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", locked_replace)
    result = MODULE.export_workbook(workbook, baseline, offload)
    assert workbook.read_bytes() == original
    output = Path(result["workbook"])
    assert output != workbook and output.is_file()
    book = openpyxl.load_workbook(output)
    assert book[MODULE.SINGLE_SHEET]["H3"].value == 100
    assert Path(result["backup"]).read_bytes() == original


def test_cli_requires_existing_workbook(experiment, tmp_path):
    _, baseline, offload = experiment([], [])
    with pytest.raises(FileNotFoundError):
        MODULE.export_workbook(tmp_path / "missing.xlsx", baseline, offload)


def test_latency_columns_use_equal_request_means(experiment):
    requests = [
        {"ttft_s": 1, "tpot_ms": 15, "itl_ms": [10, 20],
         "meta_info": {"forward_entry_time": 100, "prefill_finished_time": 100.25}},
        {"ttft_s": 3, "tpot_ms": 30, "itl_ms": [30],
         "meta_info": {"forward_entry_time": 200, "prefill_finished_time": 200.75}},
    ]
    case = ok_case("saturation", 128000, requests=requests)
    workbook, baseline, offload = experiment([], [case])
    MODULE.export_workbook(workbook, baseline, offload)
    sheet = load(workbook, MODULE.MATRIX_SHEET)
    assert [sheet.cell(4, column).value for column in (8, 9, 10)] == [500, 22.5, 22.5]
    assert sheet.cell(4, 11).value == BASE_METRICS["output_tokens_per_s"]


def test_missing_inference_start_does_not_export_client_ttft(experiment):
    case = ok_case("saturation", 128000, requests=[
        {"ttft_s": 900, "tpot_ms": 30, "itl_ms": [30], "meta_info": {}}
    ])
    workbook, baseline, offload = experiment([], [case])
    MODULE.export_workbook(workbook, baseline, offload)
    assert load(workbook, MODULE.MATRIX_SHEET).cell(4, 8).value == "未采集推理起点"
