"""Read player prompt data from Excel (.xlsx) or CSV files."""

import csv
from dataclasses import dataclass
from pathlib import Path

import openpyxl

from config import (
    COL_DETAILS,
    COL_GENDER,
    COL_ID,
    COL_PROMPT,
    CSV_DELIMITER,
    CSV_ENCODING,
    HEADER_ROW,
)


@dataclass
class PromptRow:
    id: int
    gender: str
    prompt: str
    details: str

    @property
    def full_prompt(self) -> str:
        """Combine prompt and details into the final text sent to the API."""
        parts = [self.prompt, self.details]
        return " ".join(p for p in parts if p)


def read_prompts(file_path: Path) -> list[PromptRow]:
    """Return all valid prompt rows from the given Excel or CSV file."""
    if file_path.suffix.lower() == ".csv":
        return _read_csv(file_path)
    return _read_excel(file_path)


def _read_csv(file_path: Path) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with open(file_path, newline="", encoding=CSV_ENCODING) as f:
        reader = csv.reader(f, delimiter=CSV_DELIMITER)
        next(reader)  # skip header
        for line in reader:
            if len(line) < COL_PROMPT or not line[COL_ID - 1].strip():
                continue
            prompt = line[COL_PROMPT - 1].strip()
            if not prompt:
                continue
            details = line[COL_DETAILS - 1].strip() if len(line) >= COL_DETAILS else ""
            rows.append(
                PromptRow(
                    id=int(line[COL_ID - 1]),
                    gender=line[COL_GENDER - 1].strip(),
                    prompt=prompt,
                    details=details,
                )
            )
    return rows


def _read_excel(file_path: Path) -> list[PromptRow]:
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    rows: list[PromptRow] = []

    for row in ws.iter_rows(min_row=HEADER_ROW + 1):
        row_id = row[COL_ID - 1].value
        prompt = row[COL_PROMPT - 1].value
        if row_id is None or not prompt:
            continue
        details = ""
        if len(row) >= COL_DETAILS:
            details = str(row[COL_DETAILS - 1].value or "")
        rows.append(
            PromptRow(
                id=int(row_id),
                gender=str(row[COL_GENDER - 1].value or ""),
                prompt=str(prompt),
                details=details,
            )
        )

    wb.close()
    return rows
