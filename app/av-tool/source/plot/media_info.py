import logging
from matplotlib import pyplot
from flow.data.ffprobe import *


class MediaInfoPlot:
    def __init__(self, data: FFprobeData):
        self.data = data
        self.plot_flag = False

    def show(self, width, height):
        if not self.plot_flag:
            self._plots(width, height)
        pyplot.show()

    def save(self, width, height, path):
        if self.plot_flag:
            return False
        figure = self._plots(width, height)
        figure.savefig(path)
        return True

    def _plots(self, width, height):
        x = [index for index, _ in enumerate(self.data.video.streams[0].frames)]
        y = [frame.pkt_pts for frame in self.data.video.streams[0].frames]
        logging.info(x)
        logging.info(y)
        figure, ax = pyplot.subplots(figsize=(width, height))
        ax.plot(
            x, y, linewidth=1, label="pts"
        )
        #ax.plot(
        #    self.x_extend,
        #    self.y_extend,
        #    marker="o",
        #    markersize=0,
        #    markerfacecolor="white",
        #    label=self.room,
        #)
        #ax.set_xticks(self.x_minute)
        #ax.set_xticklabels(self.x_minute_desc, fontsize=8, rotation=60)

        pyplot.xlabel("frame_index")
        pyplot.ylabel("pkt_pts")
        pyplot.title(
            f"MediaInfo/Frame",
            fontweight="bold",
            fontsize=12,
        )
        pyplot.legend()
        self.plot_flag = True
        return figure
