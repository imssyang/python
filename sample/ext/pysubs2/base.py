import pysubs2


def run():
    subs = pysubs2.load("1.ass", encoding="utf-8")
    # subs.shift(s=2.5)
    # for line in subs:
    #    line.text = "{\\be1}" + line.text
    subs.save("o.srt")


run()
