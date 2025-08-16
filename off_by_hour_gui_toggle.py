#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gooey import Gooey, GooeyParser
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd
import re
import sys

# PDF → CSV helper
from pdf_timeoff_parser import convert_timeoff_pdf_to_csv

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
HOURS = list(range(5,24))  # 5 AM..11 PM

def hour_label(h: int) -> str:
    ampm = "AM" if h < 12 else "PM"
    h12 = h if 1 <= h <= 12 else (12 if h == 12 else h - 12)
    return f"{h12} {ampm}"

def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r'[^a-z0-9\s-]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

SUFFIXES = {"jr","sr","ii","iii","iv","v"}
def split_first_last(name: str):
    n = normalize_name(name)
    parts = [p for p in n.split() if p]
    if not parts: return None, None
    if parts[-1] in SUFFIXES and len(parts) >= 2:
        parts = parts[:-1]
    if len(parts) == 1: return parts[0], parts[0]
    return parts[0], parts[-1]

def parse_time_flexible(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return None
    text = str(s).strip()
    if not text: return None
    text = text.replace(".", "").replace("–","-").replace("—","-")
    text = re.sub(r'(\d)(am|pm)', r'\1 \2', text, flags=re.I)
    text = text.lower().replace("noon","12 pm").replace("midnight","12 am")
    for fmt in ("%I:%M %p","%I %p","%H:%M","%H"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.hour*60 + dt.minute
        except Exception:
            continue
    try:
        dt = pd.to_datetime(text)
        return dt.hour*60 + dt.minute
    except Exception:
        return None

def parse_date_token(tok: str):
    m = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', str(tok))
    if not m: return None
    mm, dd, yy = m.group(1).split("/")
    mm, dd, yy = int(mm), int(dd), int(yy)
    if yy < 100: yy += 2000
    return date(yy, mm, dd)

def detect_week_from_headers(columns):
    abbr_to_full = {"Mon":"Monday","Tue":"Tuesday","Wed":"Wednesday","Thu":"Thursday","Fri":"Friday","Sat":"Saturday","Sun":"Sunday"}
    day_cols = {}
    monday_date = None
    for c in columns:
        s = str(c).strip()
        m = re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', s, flags=re.I)
        if m:
            abbr = m.group(1).title()[:3]
            full = abbr_to_full[abbr]
            day_cols[full] = c
            if full == "Monday":
                m2 = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', s)
                if m2:
                    monday_date = parse_date_token(m2.group(1))
    return monday_date, day_cols

def pick_name_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in {"employees","employee","employee name","name","team member"}:
            return c
    return df.columns[0]

# ---------- Availability-only ----------
def build_availability_counts(avail_csv: Path):
    df = pd.read_csv(avail_csv, encoding="utf-8-sig")
    name_col = pick_name_col(df)
    monday_date, day_cols = detect_week_from_headers(list(df.columns))
    counts = pd.DataFrame(0, index=[hour_label(h) for h in HOURS], columns=DAY_ORDER, dtype=int)

    for _, row in df.iterrows():
        for day in DAY_ORDER:
            col = day_cols.get(day)
            if not col: continue
            cell = str(row.get(col, "")).strip()
            if not cell: continue

            parts = re.split(r'[;,/]\s*', cell)
            ranges = []
            for part in parts:
                txt = part.strip()
                if not txt: continue
                if re.search(r'unavailable\s+all\s+day', txt, flags=re.I) or txt.lower() in {"no","unavailable"}:
                    continue
                if re.search(r'available\s+all\s+day', txt, flags=re.I) or txt.lower() in {"yes","available all day","all day","fully available"}:
                    ranges.append((0, 24*60))
                    continue
                m = re.search(r'Available\s+(.+?)\s*-\s*(.+)$', txt, flags=re.I)
                if m:
                    sstr, estr = m.group(1), m.group(2)
                else:
                    m2 = re.search(r'(\d{1,2}(:\d{2})?\s*(AM|PM))\s*-\s*(\d{1,2}(:\d{2})?\s*(AM|PM))', txt, flags=re.I)
                    if m2:
                        sstr, estr = m2.group(1), m2.group(4)
                    else:
                        continue
                smin = parse_time_flexible(sstr); emin = parse_time_flexible(estr)
                if smin is None or emin is None: continue
                if emin == 0 and smin > 0: emin = 24*60
                ranges.append((smin, emin))

            for h in HOURS:
                hs, he = h*60, (h+1)*60
                if any(s < he and e > hs for (s,e) in ranges):
                    counts.at[hour_label(h), day] += 1

    week_tag = f"{monday_date.isoformat()}_to_{(monday_date + timedelta(days=5)).isoformat()}" if monday_date else "detected_week"
    return counts, week_tag

# ---------- Time-off (intersect with availability) ----------
def load_availability_masks(avail_csv: Path):
    df = pd.read_csv(avail_csv, encoding="utf-8-sig")
    name_col = pick_name_col(df)
    monday_date, day_cols = detect_week_from_headers(list(df.columns))

    masks = {}
    names_index = {}  # norm -> (display, first, last)
    for _, row in df.iterrows():
        disp = str(row.get(name_col,"")).strip()
        if not disp: continue
        nn = normalize_name(disp)
        first, last = split_first_last(disp)
        names_index[nn] = (disp, first, last)
        masks[nn] = {d:{h:False for h in HOURS} for d in DAY_ORDER}

        for day in DAY_ORDER:
            col = day_cols.get(day)
            if not col: continue
            cell = str(row.get(col,"")).strip()
            if not cell: continue
            parts = re.split(r'[;,/]\s*', cell)
            ranges = []
            for part in parts:
                txt = part.strip()
                if not txt: continue
                if re.search(r'unavailable\s+all\s+day', txt, flags=re.I) or txt.lower() in {"no","unavailable"}:
                    continue
                if re.search(r'available\s+all\s+day', txt, flags=re.I) or txt.lower() in {"yes","available all day","all day","fully available"}:
                    ranges.append((0, 24*60 + 60))  # inclusive for intersection
                    continue
                m = re.search(r'Available\s+(.+?)\s*-\s*(.+)$', txt, flags=re.I)
                if m:
                    sstr, estr = m.group(1), m.group(2)
                else:
                    m2 = re.search(r'(\d{1,2}(:\d{2})?\s*(AM|PM))\s*-\s*(\d{1,2}(:\d{2})?\s*(AM|PM))', txt, flags=re.I)
                    if m2:
                        sstr, estr = m2.group(1), m2.group(4)
                    else:
                        continue
                smin = parse_time_flexible(sstr); emin = parse_time_flexible(estr)
                if smin is None or emin is None: continue
                emin = min(emin + 60, 24*60)  # inclusive
                ranges.append((smin, emin))
            for h in HOURS:
                hs, he = h*60, (h+1)*60
                if any(s < he and e > hs for (s,e) in ranges):
                    masks[nn][day][h] = True

    # last-name index
    by_last = {}
    for nn, (_, first, last) in names_index.items():
        if last:
            by_last.setdefault(last, set()).add(nn)

    return monday_date, masks, names_index, by_last

def extract_timeoff_events(to_csv: Path, week_start: date, week_end: date):
    # Read CSV (supports both your manual CSVs and the PDF-converted CSV with a 'Dates' column)
    df = None
    for enc in ("utf-8-sig","utf-8","cp1252"):
        try:
            df = pd.read_csv(to_csv, encoding=enc); break
        except Exception:
            continue
    if df is None:
        raise RuntimeError("Could not read time-off CSV")

    cols = {str(c).strip().lower(): c for c in df.columns}
    name_col   = next((cols[k] for k in cols if k in {"employees","employee","name","employee name","team member"}), list(df.columns)[0])
    datetime_c = next((cols[k] for k in cols if k in {"dates","date and time"} or "date and time" in k), None)

    events = []
    def add_event(raw_name: str, d: date, smin: int, emin: int):
        if week_start <= d <= week_end:
            events.append((raw_name, d, smin, emin))

    if datetime_c is not None:
        for _, r in df.iterrows():
            raw_name = str(r.get(name_col,"")).strip()
            if not raw_name: continue
            spec = str(r.get(datetime_c,"")).strip()
            if not spec: continue

            # "Mon 8/18/25 All Day"
            m1 = re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+All\s*Day$', spec, flags=re.I)
            # "Mon 8/18/25 Through Sat 8/23/25"
            m2 = re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+Through\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\d{1,2}/\d{1,2}/\d{2,4})$', spec, flags=re.I)

            if m1:
                d = parse_date_token(m1.group(2))
                if d: add_event(raw_name, d, 0, 24*60 + 60)
                continue
            if m2:
                d1 = parse_date_token(m2.group(2))
                d2 = parse_date_token(m2.group(4))
                if d1 and d2:
                    cur = max(min(d1,d2), week_start)
                    fin = min(max(d1,d2), week_end)
                    while cur <= fin:
                        add_event(raw_name, cur, 0, 24*60 + 60)
                        cur += timedelta(days=1)
                continue
            # Fallback: if it's a single date without keywords, treat as all-day
            m3 = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', spec)
            if m3:
                d = parse_date_token(m3.group(1))
                if d: add_event(raw_name, d, 0, 24*60 + 60)

    return events

def build_timeoff_sum(avail_csv: Path, timeoff_csv: Path, week_monday: date|None):
    detected_monday, masks, names_index, by_last = load_availability_masks(avail_csv)
    if not week_monday:
        if not detected_monday:
            raise SystemExit("Could not detect the week from Availability headers. Enter Week Monday manually.")
        week_monday = detected_monday
    week_end = week_monday + timedelta(days=5)

    events = extract_timeoff_events(timeoff_csv, week_monday, week_end)

    # map names by last name + first initial
    def map_to_availability(foh_name: str):
        # Normalize split
        def normalize_name_local(s: str) -> str:
            s = str(s).strip().lower()
            s = re.sub(r'[^a-z0-9\s-]', '', s)
            s = re.sub(r'\s+', ' ', s)
            return s
        def split_first_last_local(name: str):
            SUFFIXES = {"jr","sr","ii","iii","iv","v"}
            n = normalize_name_local(name)
            parts = [p for p in n.split() if p]
            if not parts: return None, None
            if parts[-1] in SUFFIXES and len(parts) >= 2:
                parts = parts[:-1]
            if len(parts) == 1: return parts[0], parts[0]
            return parts[0], parts[-1]

        f_first, f_last = split_first_last_local(foh_name)
        cands = list(by_last.get(f_last, [])) if f_last else []
        if not cands: return None, "no_match"
        if len(cands) == 1: return cands[0], "matched_last"
        fin = (f_first or "")[:1]
        cands_initial = [c for c in cands if (names_index[c][1] or "").startswith(fin)]
        if len(cands_initial) == 1: return cands_initial[0], "matched_last_first_initial"
        cands_exact = [c for c in cands if (names_index[c][1] or "") == f_first]
        if len(cands_exact) == 1: return cands_exact[0], "matched_last_first_exact"
        return None, "ambiguous"

    from collections import defaultdict
    person_day_ranges = defaultdict(lambda: {d:[] for d in DAY_ORDER})
    mappings = []
    name_to_avail = {}

    for raw_name, d, smin, emin in events:
        if raw_name not in name_to_avail:
            mapped, reason = map_to_availability(raw_name)
            name_to_avail[raw_name] = (mapped, reason)
        mapped, reason = name_to_avail[raw_name]
        mappings.append({"FOH Name": raw_name, "Mapped Availability Norm": mapped or "", "Match Type": reason})
        person_day_ranges[raw_name][d.strftime("%A")].append((smin, emin))

    sum_df = pd.DataFrame(0, index=[hour_label(h) for h in HOURS], columns=DAY_ORDER, dtype=int)
    for raw_name, day_ranges in person_day_ranges.items():
        mapped, _ = name_to_avail.get(raw_name, (None,"no_match"))
        if not mapped or mapped not in masks: continue
        amask = masks[mapped]
        for day in DAY_ORDER:
            ranges = day_ranges.get(day, [])
            if not ranges: continue
            for h in HOURS:
                hs, he = h*60, (h+1)*60
                off_overlap = any(smin < he and emin > hs for (smin, emin) in ranges)
                if off_overlap and amask[day].get(h, False):
                    sum_df.at[hour_label(h), day] += 1

    return sum_df, mappings, week_monday, week_end

@Gooey(program_name="Weekly Hourly Reports",
       program_description="Toggle between Availability-by-hour or Time-off-by-hour",
       default_size=(800, 600), clear_before_run=True)
def main():
    parser = GooeyParser()
    subparsers = parser.add_subparsers(dest="mode", help="Choose a report type")

    p_av = subparsers.add_parser("Availability", help="Availability-by-hour (Availability CSV only)")
    p_av.add_argument("--availability", required=True, widget="FileChooser", help="Availability CSV")
    p_av.add_argument("--outdir", default="./out", widget="DirChooser", help="Output folder")

    p_to = subparsers.add_parser("TimeOff", help="Time-off-by-hour (Availability + Time-Off CSV/PDF)")
    p_to.add_argument("--availability", required=True, widget="FileChooser", help="Availability CSV")
    p_to.add_argument("--timeoff", required=True, widget="FileChooser", help="Time-Off file (CSV or PDF report)")
    p_to.add_argument("--outdir", default="./out", widget="DirChooser", help="Output folder")
    p_to.add_argument("--week-monday", help="YYYY-MM-DD (optional; auto-read from Availability if omitted)")

    args = parser.parse_args()
    outdir = Path(getattr(args, "outdir", "./out")); outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "Availability":
        counts, week_tag = build_availability_counts(Path(args.availability))
        xlsx = outdir / f"AVAILABILITY_COUNTS_{week_tag}.xlsx"
        csv  = outdir / f"AVAILABILITY_COUNTS_{week_tag}.csv"
        with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
            counts.to_excel(writer, sheet_name="Counts (Mon–Sat 5A–11P)", index=True)
        counts.to_csv(csv, index=True)
        print(f"\nSaved:\n- {xlsx}\n- {csv}\n")

    elif args.mode == "TimeOff":
        week_monday = None
        if getattr(args, "week_monday", None):
            week_monday = datetime.strptime(args.week_monday, "%Y-%m-%d").date()

        # Accept PDFs too: convert to CSV in the output folder, then proceed
        to_path = Path(args.timeoff)
        if to_path.suffix.lower() == ".pdf":
            to_path = convert_timeoff_pdf_to_csv(to_path, outdir / "TimeOff_from_PDF.csv")

        sum_df, mappings, wk_start, wk_end = build_timeoff_sum(Path(args.availability), to_path, week_monday)
        week_tag = f"{wk_start.isoformat()}_to_{wk_end.isoformat()}"
        xlsx = outdir / f"OFF_by_hour_SUM_{week_tag}.xlsx"
        csv  = outdir / f"OFF_by_hour_SUM_{week_tag}.csv"
        with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
            sum_df.to_excel(writer, sheet_name="Sum", index=True)
        sum_df.to_csv(csv, index=True)
        map_csv = outdir / f"Name_Mapping_{wk_start.isoformat()}.csv"
        pd.DataFrame(mappings).to_csv(map_csv, index=False)
        print(f"\nSaved:\n- {xlsx}\n- {csv}\n- {map_csv}\n")

    else:
        print("Please pick a mode (Availability or TimeOff).")
        sys.exit(2)

if __name__ == "__main__":
    main()
