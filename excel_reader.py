"""Read prompt rows from an Excel spreadsheet."""

from dataclasses import dataclass
from pathlib import Path

import openpyxl

from config import COL_DETAILS, COL_GENDER, COL_ID, COL_PROMPT, HEADER_ROW


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


def read_prompts(excel_path: Path) -> list[PromptRow]:
    """Return all valid prompt rows from the given Excel file."""
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active
    rows: list[PromptRow] = []

    for row in ws.iter_rows(min_row=HEADER_ROW + 1):
        row_id = row[COL_ID - 1].value
        prompt = row[COL_PROMPT - 1].value
        if row_id is None or not prompt:
            continue
        rows.append(
            PromptRow(
                id=int(row_id),
                gender=str(row[COL_GENDER - 1].value or ""),
                prompt=str(prompt),
                details=str(row[COL_DETAILS - 1].value or ""),
            )
        )

    wb.close()
    return rows
