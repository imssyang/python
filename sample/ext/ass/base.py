import ass


def run():
    with open("1.ass", encoding="utf_8_sig") as rf:
        doc = ass.parse(rf)
        print(doc.info)
        print(doc.styles)
        # print(doc.sections.keys())
        print(doc.events[0])
        with open("o.ass", "w", encoding="utf_8_sig") as wf:
            doc.dump_file(wf)


run()
