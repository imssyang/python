import csv
import os


def write(path: str):
    with open(path, "w", encoding="utf-8") as file:
        fields = ["algo1", "algo2", "start_time", "end_time"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        names = os.listdir("hanbo")
        for name in names:
            if "tune0.mp4" in name:
                name2 = name.replace("tune0", "tune1")
                row = [
                    dict(
                        algo1=f"upos://sucaiboss/hanbo/{name}",
                        algo2=f"upos://sucaiboss/hanbo/{name2}",
                        start_time=20000,
                        end_time=30000,
                    )
                ]
                writer.writerows(row)


write("hanbo.csv")
