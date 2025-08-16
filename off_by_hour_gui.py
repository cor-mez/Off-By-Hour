from gooey import Gooey, GooeyParser
import subprocess, sys, os

@Gooey(
    program_name="OFF-by-hour Weekly",
    program_description="Pick Availability + Time-Off CSVs and get the Sum table",
    default_size=(760, 560),
    required_cols=1,
    clear_before_run=True
)
def main():
    p = GooeyParser()
    p.add_argument('--availability', required=True, widget='FileChooser', help='Availability CSV')
    p.add_argument('--timeoff',      required=True, widget='FileChooser', help='Time-Off CSV')
    p.add_argument('--outdir',       default='./out', widget='DirChooser', help='Output folder')
    p.add_argument('--week-monday',  help='YYYY-MM-DD (optional; auto-detected if omitted)')
    p.add_argument('--include-sunday', action='store_true', help='Include Sunday column')
    p.add_argument('--status-filter', nargs='*', help='Only include these time-off statuses (e.g. Approved)')
    p.add_argument('--alias-csv', widget='FileChooser', help='Optional alias CSV (foh_name,availability_name)')
    p.add_argument('--per-person-xlsx', action='store_true', help='Also write per-person 1/0 sheets')
    args = p.parse_args()

    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), 'off_by_hour_weekly.py'),
        '--availability', args.availability,
        '--timeoff', args.timeoff,
        '--outdir', args.outdir
    ]
    if args.week_monday:       cmd += ['--week-monday', args.week_monday]
    if args.include_sunday:    cmd += ['--include-sunday']
    if args.status_filter:     cmd += ['--status-filter', *args.status_filter]
    if args.alias_csv:         cmd += ['--alias-csv', args.alias_csv]
    if args.per_person_xlsx:   cmd += ['--per-person-xlsx']

    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
