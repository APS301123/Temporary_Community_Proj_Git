from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import gspread

from src.detector import DetectionRecord

DETECTIONS_HEADERS = [
    "timestamp_utc",
    "image_id",
    "source_filename",
    "group",
    "subgroup",
    "descriptive_subgroup",
    "model_label",
    "count",
    "confidence",
    "color",
    "size_bucket",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]

SUMMARY_HEADERS = [
    "group",
    "descriptive_subgroup",
    "total_items",
    "last_updated_utc",
]

GROUP_TOTAL_HEADERS = [
    "group",
    "total_items",
    "last_updated_utc",
]


class SheetsSync:
    def __init__(
        self,
        service_account_json: str,
        sheet_id: str,
        detections_tab: str,
        summary_tab: str,
        group_totals_tab: str,
    ) -> None:
        client = gspread.service_account(filename=service_account_json)
        self.sheet = client.open_by_key(sheet_id)
        self.detections_ws = self._ensure_worksheet(detections_tab, DETECTIONS_HEADERS)
        self.summary_ws = self._ensure_worksheet(summary_tab, SUMMARY_HEADERS)
        self.group_totals_ws = self._ensure_worksheet(group_totals_tab, GROUP_TOTAL_HEADERS)

    def _ensure_worksheet(self, title: str, headers: list[str]):
        try:
            worksheet = self.sheet.worksheet(title)
        except gspread.WorksheetNotFound:
            worksheet = self.sheet.add_worksheet(title=title, rows=1000, cols=26)
            worksheet.append_row(headers)
            return worksheet

        current_headers = worksheet.row_values(1)
        if current_headers != headers:
            worksheet.clear()
            worksheet.append_row(headers)
        return worksheet

    def record_submission(
        self, image_id: str, source_filename: str, detections: list[DetectionRecord]
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        if detections:
            rows = []
            for item in detections:
                x1, y1, x2, y2 = item.bbox
                rows.append(
                    [
                        timestamp,
                        image_id,
                        source_filename,
                        item.group,
                        item.subgroup,
                        item.descriptive_subgroup,
                        item.label,
                        1,
                        round(item.confidence, 4),
                        ", ".join(item.color),
                        item.size_bucket,
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )
            self.detections_ws.append_rows(rows, value_input_option="RAW")
        else:
            self.detections_ws.append_row(
                [
                    timestamp,
                    image_id,
                    source_filename,
                    "No Detection",
                    "No Detection",
                    "No Detection",
                    "",
                    0,
                    0,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
                value_input_option="RAW",
            )

        self._upsert_summary(detections)
        self._upsert_group_totals(detections)

    def _upsert_summary(self, detections: list[DetectionRecord]) -> None:
        if not detections:
            return

        now = datetime.now(timezone.utc).isoformat()
        existing = self.summary_ws.get_all_records()
        index_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for idx, row in enumerate(existing, start=2):
            key = (str(row["group"]), str(row["descriptive_subgroup"]))
            row["_row_index"] = idx
            index_by_key[key] = row

        grouped_counts: dict[tuple[str, str], int] = {}
        for detection in detections:
            key = (detection.group, detection.descriptive_subgroup)
            grouped_counts[key] = grouped_counts.get(key, 0) + 1

        rows_to_append: list[list[Any]] = []
        updates: list[tuple[int, int, list[Any]]] = []
        for (group, descriptive_subgroup), increment in grouped_counts.items():
            existing_row = index_by_key.get((group, descriptive_subgroup))
            if existing_row is None:
                rows_to_append.append([group, descriptive_subgroup, increment, now])
                continue

            new_total = int(existing_row["total_items"]) + increment
            row_idx = int(existing_row["_row_index"])
            updates.append((row_idx, 3, [[new_total]]))
            updates.append((row_idx, 4, [[now]]))

        if rows_to_append:
            self.summary_ws.append_rows(rows_to_append, value_input_option="RAW")
        for row, col, values in updates:
            self.summary_ws.update_cell(row, col, values[0][0])

    def _upsert_group_totals(self, detections: list[DetectionRecord]) -> None:
        if not detections:
            return

        now = datetime.now(timezone.utc).isoformat()
        existing = self.group_totals_ws.get_all_records()
        index_by_group: dict[str, dict[str, Any]] = {}
        for idx, row in enumerate(existing, start=2):
            group = str(row["group"])
            row["_row_index"] = idx
            index_by_group[group] = row

        group_counts: dict[str, int] = {}
        for detection in detections:
            group_counts[detection.group] = group_counts.get(detection.group, 0) + 1

        rows_to_append: list[list[Any]] = []
        updates: list[tuple[int, int, Any]] = []
        for group, increment in group_counts.items():
            existing_row = index_by_group.get(group)
            if existing_row is None:
                rows_to_append.append([group, increment, now])
                continue
            new_total = int(existing_row["total_items"]) + increment
            row_idx = int(existing_row["_row_index"])
            updates.append((row_idx, 2, new_total))
            updates.append((row_idx, 3, now))

        if rows_to_append:
            self.group_totals_ws.append_rows(rows_to_append, value_input_option="RAW")
        for row, col, value in updates:
            self.group_totals_ws.update_cell(row, col, value)

