#!/usr/bin/env python3
"""Push a pytest JUnit XML report to Google Sheets.

Writes three tabs on the target spreadsheet:
  - "Runs"          one row per CI run (summary counts + duration)
  - "Test Results"  one row per test per run
  - "Failures"      one row per failing/erroring test per run, with message

Reads credentials from GOOGLE_SHEETS_CREDENTIALS (service-account JSON,
as a string) and the target sheet from GOOGLE_SHEET_ID. Both are read
from the environment so no secret ever touches argv or a file on disk.
Skips silently (exit 0) if either is unset, so this step stays a no-op
on forked-repo PRs that don't have access to repo secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

RUNS_HEADER = [
    "timestamp", "commit", "branch", "run_url",
    "total", "passed", "failed", "errors", "skipped", "duration_s",
]
RESULTS_HEADER = [
    "timestamp", "commit", "branch", "classname", "test", "outcome", "duration_s",
]
FAILURES_HEADER = [
    "timestamp", "commit", "branch", "classname", "test", "outcome", "message", "run_url",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--junit-xml", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--run-url", required=True)
    return p.parse_args()


def iter_testcases(xml_path: str):
    root = ET.parse(xml_path).getroot()
    suites = root.iter("testsuite") if root.tag == "testsuites" else [root]
    for suite in suites:
        yield from suite.iter("testcase")


def outcome_of(testcase: ET.Element) -> tuple[str, str | None]:
    for tag, outcome in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
        node = testcase.find(tag)
        if node is not None:
            message = node.get("message") or (node.text or "").strip()
            return outcome, message[:2000]
    return "passed", None


def build_rows(junit_xml: str, commit: str, branch: str, run_url: str):
    ts = datetime.now(timezone.utc).isoformat()
    short_commit = commit[:8]

    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    total_duration = 0.0
    result_rows: list[list[str]] = []
    failure_rows: list[list[str]] = []

    for tc in iter_testcases(junit_xml):
        name = tc.get("name", "")
        classname = tc.get("classname", "")
        duration = float(tc.get("time", "0") or 0)
        total_duration += duration
        outcome, message = outcome_of(tc)
        counts[outcome] = counts.get(outcome, 0) + 1

        result_rows.append([ts, short_commit, branch, classname, name, outcome, f"{duration:.3f}"])
        if outcome in ("failed", "error"):
            failure_rows.append([ts, short_commit, branch, classname, name, outcome, message or "", run_url])

    total = sum(counts.values())
    summary_row = [
        ts, short_commit, branch, run_url,
        str(total), str(counts["passed"]), str(counts["failed"]),
        str(counts["error"]), str(counts["skipped"]), f"{total_duration:.2f}",
    ]
    return summary_row, result_rows, failure_rows


def get_or_create_worksheet(spreadsheet, title: str, header: list[str]):
    try:
        ws = spreadsheet.worksheet(title)
    except Exception:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=len(header))
        ws.append_row(header)
        return ws
    if not ws.row_values(1):
        ws.append_row(header)
    return ws


def main() -> int:
    args = parse_args()

    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not creds_json or not sheet_id:
        print("GOOGLE_SHEETS_CREDENTIALS or GOOGLE_SHEET_ID not set — skipping sheet report.")
        return 0

    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(sheet_id)

    summary_row, result_rows, failure_rows = build_rows(
        args.junit_xml, args.commit, args.branch, args.run_url,
    )

    runs_ws = get_or_create_worksheet(spreadsheet, "Runs", RUNS_HEADER)
    runs_ws.append_row(summary_row)

    if result_rows:
        results_ws = get_or_create_worksheet(spreadsheet, "Test Results", RESULTS_HEADER)
        results_ws.append_rows(result_rows)

    if failure_rows:
        failures_ws = get_or_create_worksheet(spreadsheet, "Failures", FAILURES_HEADER)
        failures_ws.append_rows(failure_rows)

    print(f"Reported {len(result_rows)} test results ({len(failure_rows)} failures) to sheet {sheet_id}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
