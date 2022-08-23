# -*- coding: utf-8 -*-
import csv
import numpy
import time
from datetime import datetime
from matplotlib import pyplot
from scipy.interpolate import Rbf


class BarrageData:
    def __init__(
        self, path: str, room: str, from_time: str, to_time: str, weight: float
    ):
        self.path = path
        self.room = room
        self.from_time = from_time
        self.to_time = to_time
        self.weight = weight
        self.middle_threshold = 0
        self.weight_threshold = 0
        self.x_minute = []
        self.x_minute_desc = []
        self.x_desc = []
        self.x = []
        self.y = []
        self.x_extend = []
        self.y_extend = []
        self.y_weight = []
        self.p_indexs = []
        self.plot_flag = False
        self._read()
        self._calc()

    def show(self, width=15, height=6):
        if not self.plot_flag:
            self._plots(width, height)
        pyplot.show()

    def save(self, path: str, width=15, height=6):
        if self.plot_flag:
            return False
        figure = self._plots(width, height)
        figure.savefig(path)
        return True

    def fragments(self):
        fragments = []
        x_start = self.x_extend[0]
        x_end = self.x_extend[-1]
        for index in self.p_indexs:
            if self._ascent(index) is True:
                x_start = self.x_extend[index]
            elif self._ascent(index) is False:
                x_end = self.x_extend[index]
                if x_end > x_start:
                    fragments.append((x_start, x_end))
        return fragments

    def _plots(self, width, height):
        figure, ax = pyplot.subplots(figsize=(width, height))
        ax.plot(
            self.x, [self.middle_threshold] * len(self.x), linewidth=1.5, label="middle"
        )
        ax.plot(
            self.x,
            [self.weight_threshold] * len(self.x),
            linewidth=1.5,
            label=f"weight({self.weight})",
        )
        ax.plot(
            self.x_extend,
            self.y_extend,
            marker="o",
            markersize=0,
            markerfacecolor="white",
            label=self.room,
        )
        ax.plot(
            self.x_extend[self.p_indexs],
            self.y_weight[self.p_indexs],
            "bo",
            markersize=3,
        )
        ax.set_xticks(self.x_minute)
        ax.set_xticklabels(self.x_minute_desc, fontsize=8, rotation=60)

        for index in self.p_indexs:
            x_time = datetime.fromtimestamp(self.x_extend[index])
            x_time_text = x_time.strftime("%H:%M:%S")
            if self._ascent(index) is True:
                pyplot.annotate(
                    x_time_text,
                    (self.x_extend[index], self.y_weight[index]),
                    fontsize=8,
                    rotation=60,
                )
            elif self._ascent(index) is False:
                pyplot.annotate(
                    x_time_text,
                    xy=(self.x_extend[index], self.y_weight[index]),
                    verticalalignment="top",
                    fontsize=8,
                    rotation=-60,
                )

        pyplot.fill_between(
            x=self.x_extend,
            y1=self.y_extend,
            y2=self.y_weight,
            where=self.y_extend > self.weight_threshold,
            color="b",
            alpha=0.2,
        )
        pyplot.xlabel("TIME")
        pyplot.ylabel("NUM")
        pyplot.title(
            f"Barrages [{self.from_time}~{self.to_time}]",
            fontweight="bold",
            fontsize=12,
        )
        pyplot.legend()
        self.plot_flag = True
        return figure

    def _read(self):
        from_st_time = time.strptime(self.from_time, "%Y-%m-%d %H:%M:%S")
        from_raw_time = time.mktime(from_st_time)
        to_st_time = time.strptime(self.to_time, "%Y-%m-%d %H:%M:%S")
        to_raw_time = time.mktime(to_st_time)

        csv_items = list()
        with open(self.path, "r") as file:
            lines = csv.reader(file, delimiter=",")
            for line in lines:
                room_item = line[0]
                if room_item != self.room:
                    continue

                time_item = line[1]
                st_time = time.strptime(time_item, "%m/%d/%Y %H:%M")
                raw_time = time.mktime(st_time)
                if from_raw_time > raw_time or raw_time > to_raw_time:
                    continue

                value_item = line[2]
                if not value_item.isnumeric():
                    continue

                csv_items.append(
                    dict(
                        room=room_item,
                        time_desc=time_item[-5:],
                        timestamp=raw_time,
                        value=int(value_item),
                    )
                )

        csv_items.sort(key=lambda item: item["timestamp"])

        rcs_x_desc = []
        rcs_x = []
        rcs_y = []
        for item in csv_items:
            if 0 < len(rcs_x):
                last_x_desc = rcs_x_desc[-1]
                last_x = rcs_x[-1]
                if last_x != item["timestamp"]:
                    self.x_minute_desc.append(last_x_desc)
                    self.x_minute.append(last_x)

                    cache_nums = len(rcs_x)
                    threshold = 60 / cache_nums
                    for index in range(cache_nums):
                        second = int(threshold * index)
                        self.x_desc.append(rcs_x_desc[index] + f":{second:02}")
                        self.x.append(rcs_x[index] + second)
                        self.y.append(rcs_y[index])
                    rcs_x_desc.clear()
                    rcs_x.clear()
                    rcs_y.clear()
            rcs_x_desc.append(item["time_desc"])
            rcs_x.append(item["timestamp"])
            rcs_y.append(item["value"])

    def _calc(self):
        if 0 >= len(self.y):
            return

        y = sorted(self.y)
        self.middle_threshold = y[len(y) // 2] + 1
        self.weight_threshold = self.weight * self.middle_threshold

        x_new = numpy.array(self.x)
        y_new = numpy.array(self.y)
        rbf = Rbf(x_new, y_new, function="thin_plate", epsilon=0.1, smooth=0.0001)
        self.x_extend = numpy.linspace(x_new.min(), x_new.max(), 10000)
        self.y_extend = rbf(self.x_extend)
        self.y_weight = numpy.array([self.weight_threshold] * len(self.y_extend))
        self.p_indexs = numpy.argwhere(
            numpy.diff(numpy.sign(self.y_extend - self.y_weight))
        ).flatten()

    def _ascent(self, index):
        y_index = self.y_extend[index]
        for i in range(10):
            next_index = index + i + 1
            if self.y_extend[next_index] == y_index:
                continue
            elif self.y_extend[next_index] > y_index:
                return True
            elif self.y_extend[next_index] < y_index:
                return False
        return None
