# A20: summarise a log from a ttsim built with ../patches/a20-counters.patch: NoC totals and Tensix opcode counts
# (names from the generated tensix_decode.h of $TTSIM_HOME; run make.py once there). Usage: python a20_summary.py run.log ...
import os, re, sys
names = {}
for line in open(os.environ.get('TTSIM_HOME', os.path.expanduser('~/ttsim')) + '/src/_out/wh/tensix_decode.h'):
    m = re.match(r'#define TENSIX_OPCODE_(\w+) 0x([0-9A-Fa-f]+)$', line.strip())
    if m: names[int(m.group(2), 16)] = m.group(1)
for path in sys.argv[1:]:
    lines = [l.split('] ', 1)[1].strip() for l in open(path) if re.match(r'^\[[1-9]\d*\] A20 ', l)]
    print('==', path.split('/')[-1])
    for l in lines:
        if l.startswith('A20 noc') and 'cmds=0 ' not in l: print('  ' + l[4:])
        if l.startswith('A20 tensix'): print('  ' + l[4:])
    ops = {}
    for l in lines:
        m = re.match(r'A20 op T(\d) 0x([0-9a-f]+) (\d+)', l)
        if m: ops[(int(m.group(1)), names.get(int(m.group(2), 16), m.group(2)))] = int(m.group(3))
    sfpu = sum(v for (t, n), v in ops.items() if n.startswith('SFP'))
    keys = ['MVMUL', 'ELWADD', 'ELWMUL', 'UNPACR', 'PACR']
    print('  ops: ' + ' '.join(f'{k}={sum(v for (t, n), v in ops.items() if n == k)}' for k in keys) + f' SFPU*={sfpu}')
