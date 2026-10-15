import re
import sys

tmpl, frag, out = sys.argv[1:4]
parts = {}
cur = None
for line in open(frag):
    m = re.match(r'^<<(\w+)>>$', line.strip())
    if m:
        cur = m.group(1)
        parts[cur] = []
    elif cur:
        parts[cur].append(line)
txt = open(tmpl).read()
for k, v in parts.items():
    txt = txt.replace(f'<<{k}>>', ''.join(v).strip())
assert '<<' not in txt, 'unfilled placeholder'
open(out, 'w').write(txt)
print('wrote', out, len(txt))
