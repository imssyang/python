
if "123".isdigit():
    print(int("123"))
if not 'abc'.isdigit():
    print("abc")

hero = {}
hero.update(
    {
                'id': hero.get('id', 0),
                'title': hero.get('title', ""),
                'prob': hero.get('prob', ""),
                'name': hero.get('name', ""),
                'avlie': 0,
    }
)
print(f"{hero}")

def aaa(old, new):
    a,b = old
    c,d = new
    print(a, b, c, d)
    new = old
    return new

old = [1, 2]
new = [3, 4]
new = aaa(old, new)
print(f"new: {new}")

print(len((1,)))
print(len((1,2,3)))
print(enumerate(['a', 'b']))                 # <enumerate object at 0x7faf6a418a80>
print(list(enumerate(['a', 'b'])))           # [(0, 'a'), (1, 'b')]
print(list(enumerate(['a', 'b'], start=1)))  # [(1, 'a'), (2, 'b')]
for i, value in enumerate(['a', 'b']):       # 0 a 1 b
    print(i, value)


print(sum((0.12, 4.25)))

#print(int('nash'.split('_')[-1]))
