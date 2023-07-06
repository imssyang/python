import re


def match(x):
    print(type(x), x, x.group(0), x.group(1), x.group(2))
    return f"in:{x.group(1)} pkt_pts:{x.group(2)}123"


aa = "version:1 width:1080 height:1920"
a = "in:5 pkt_pts:1536 cnt:2 8,0,354,1080,1538 1,244,572,386,524"
p = re.compile(r"in:([0-9]+) pkt_pts:([0-9]+)")
b = p.sub(lambda x: f"in:{x.group(1)} pkt_pts:{x.group(2)}123", a)
# b = p.sub(match, a)
print(b)

p1 = re.compile(r"in:([0-9]+)")
m1 = p1.match(a)
print(m1, m1.group(0), m1.group(1))

p2 = re.compile(r"pkt_pts:([0-9]+)")
m2 = p2.search(a)
print(m2, m2.group(0), m2.group(1))
