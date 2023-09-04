import pysubs2

subs = pysubs2.load("01.edit.srt", encoding="utf-8")
subs.shift(s=2.5)
for line in subs:
    line.text = "{\\be1}" + line.text
subs.save("01.edit.2.ass")
