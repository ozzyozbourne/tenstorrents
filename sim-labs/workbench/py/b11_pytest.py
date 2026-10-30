"""B11: run a slice of tt-metal's pytest suite under ttsim and survive simulator aborts.

    python b11_pytest.py <test file or dir> [extra pytest args, e.g. -k "not FLOAT32"]

ttsim reports an unsupported or undefined instruction by printing "[clock] ERROR: ..." and calling
_Exit(1), which kills the whole pytest process. This driver records each test's outcome, and when
pytest dies it restarts it with every finished test and the test that killed it deselected. At the
end it prints the counts and the ttsim message behind each abort.
"""
import re, subprocess, sys, collections

args = sys.argv[1:]
if not args:
    raise SystemExit(__doc__)
result = re.compile(r"(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)")  # may be glued to the next log line
simerr = re.compile(r"^\[\d+\] ERROR: (.*)$", re.M)
base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "--import-mode=importlib"]
# Collect the node ids first: some contain spaces, so they cannot be cut out of the log with a regex.
ids = [l for l in subprocess.run(base + ["--collect-only", "-q"] + args, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True).stdout.splitlines() if "::" in l]
by_func = collections.defaultdict(list)
for i in ids:
    by_func[i.split("[")[0]].append(i)
def node_at(line):                                   # "-v" prints the node id before running the test
    k = line.find("tests/")
    if k < 0 or "::" not in line:
        return None, 0
    rest = line[k:]
    for i in by_func.get(re.split(r"[\[ ]", rest, 1)[0], []):
        if rest == i or rest.startswith(i + " "):
            return i, k + len(i)
    return None, 0
print("collected %d tests" % len(ids), flush=True)
outcome = {}                                         # node id -> PASSED / FAILED / SKIPPED / ...
aborts = []                                          # (node id, ttsim message)
reasons = []
runs = 0
while True:
    # Deselect the tests that passed or aborted. Skips and failures run again, so that the final run's
    # "-rfEs" summary lists every skip reason and failure message (skips cost nothing; failures are rare).
    skip = [t for t, o in outcome.items() if o == "PASSED"] + [a[0] for a in aborts]
    cmd = base + ["-v", "-s", "-rfEs"] + [x for d in skip for x in ("--deselect", d)] + args
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    runs += 1
    current = None
    reasons = []
    for line in p.stdout.splitlines():
        if line.startswith("SKIPPED [") or (line.startswith(("FAILED ", "ERROR ")) and "::" in line and " - " in line):
            reasons.append(line)
            continue
        node, end = node_at(line)
        if node:
            current = node
        r = result.match(line[end:].lstrip() if node else line)  # outcome follows the id or starts a line
        if r and current:
            outcome[current] = r.group(1)
            current = None
    finished = re.search(r"=+ .*\d+ (passed|failed|skipped|deselected|error).* in [\d.]+s", p.stdout)
    if finished and current is None:
        break
    if not current:
        sys.exit("pytest died outside a test (exit %d):\n%s" % (p.returncode, p.stdout[-3000:]))
    errs = simerr.findall(p.stdout)
    aborts.append((current, errs[-1] if errs else "exit code %d, no ttsim ERROR line" % p.returncode))
    print("ABORT %s\n      %s" % aborts[-1], flush=True)

counts = collections.Counter(outcome.values())
print("pytest runs: %d" % runs)
print("counts: %s ABORTED=%d" % (" ".join("%s=%d" % kv for kv in sorted(counts.items())), len(aborts)))
for line in reasons:                                 # pytest's -rfEs summary of the final run
    print("  " + re.sub(r"\S*?(tests/)", r"\1", line, count=1)[:220])
for reason, n in collections.Counter(e for _, e in aborts).most_common():
    print("  %3d x ABORT %s" % (n, reason))
