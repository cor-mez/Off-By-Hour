from pathlib import Path
import re
import pandas as pd

# Recognizers for tokens we expect line-by-line from the PDF
DOW_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", flags=re.I)
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
THROUGH_RE = re.compile(r"^Through$", flags=re.I)
ALLDAY_RE = re.compile(r"^All Day$", flags=re.I)
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .,'\-]+(?:\s[A-Za-z .,'\-]+)*$")

HEADER_NOISE = {
    "time off & request report","time off needed","approved",
    "hot schedules - online employee scheduling, restaurant schedules, employee scheduling software and workforce communication"
}

def _lines_from_pdf(path: Path):
    lines = []
    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                for raw in txt.splitlines():
                    s = re.sub(r"\s+", " ", raw).strip()
                    if s: lines.append(s)
        if lines: return lines
    except Exception:
        pass
    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        for p in reader.pages:
            txt = p.extract_text() or ""
            for raw in txt.splitlines():
                s = re.sub(r"\s+", " ", raw).strip()
                if s: lines.append(s)
    except Exception:
        pass
    return lines

def _is_headerish(s: str) -> bool:
    sl = s.lower()
    if sl in HEADER_NOISE: return True
    if sl.startswith("approved"): return True
    if re.search(r"\d{1,2}:\d{2}\s*(am|pm)", sl): return True  # timestamp lines
    if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4},", sl): return True  # footer like "8/16/25, 2:38 PM"
    return False

def parse_timeoff_pdf_to_df(pdf_path: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Employees, Dates
    Dates look like:
      - "Mon 8/18/25 Through Sat 8/23/25"
      - "Wed 8/27/25 All Day"
    Handles multi-line PDFs where DOW/Date/Through are split across lines.
    """
    lines = _lines_from_pdf(pdf_path)
    entries = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or _is_headerish(line):
            i += 1; continue

        # Name lines are not DOW/DATE/Through/All Day and look like a person
        if NAME_RE.match(line) and not (DOW_RE.match(line) or DATE_RE.match(line) or THROUGH_RE.match(line) or ALLDAY_RE.match(line)):
            if line.lower() in {"name","date and time","name date and time"}:
                i += 1; continue

            # Walk forward: DOW1, DATE1, [All Day] OR [Through DOW2 DATE2]
            dow1 = d1 = dow2 = d2 = None
            allday = False
            j = i + 1
            while j < len(lines):
                tok = lines[j].strip()
                if not tok or _is_headerish(tok):
                    j += 1; continue
                if dow1 is None and DOW_RE.match(tok):
                    dow1 = tok.split()[0].title(); j += 1; continue
                if d1 is None and DATE_RE.match(tok):
                    d1 = tok; j += 1; continue
                if ALLDAY_RE.match(tok):
                    allday = True; j += 1; break
                if THROUGH_RE.match(tok):
                    j += 1
                    while j < len(lines) and _is_headerish(lines[j]): j += 1
                    if j < len(lines) and DOW_RE.match(lines[j]):
                        dow2 = lines[j].split()[0].title(); j += 1
                    while j < len(lines) and _is_headerish(lines[j]): j += 1
                    if j < len(lines) and DATE_RE.match(lines[j]):
                        d2 = lines[j]; j += 1
                    break
                # If a new name appears before finishing, stop this record
                if NAME_RE.match(tok) and not DOW_RE.match(tok) and not DATE_RE.match(tok):
                    break
                j += 1

            if dow1 and d1 and (allday or (dow2 and d2)):
                dates = f"{dow1} {d1} All Day" if allday else f"{dow1} {d1} Through {dow2} {d2}"
                entries.append((line, dates))
                i = j
                continue

        i += 1

    df = pd.DataFrame(entries, columns=["Employees","Dates"])
    return df

def convert_timeoff_pdf_to_csv(pdf_path: Path, out_csv_path: Path) -> Path:
    df = parse_timeoff_pdf_to_df(pdf_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path
