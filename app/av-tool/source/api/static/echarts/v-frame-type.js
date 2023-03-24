function getVFrameTypeOption(chart, partOption) {
  chart.on('magictypechanged', function (params) {
    var series = params.newOption.series;
    if (series.length > 0) {
      if (params.currentType == "line") {
        series.map((item, i) => { item.stack = item.name; });
      } else if (params.currentType == "bar") {
        series.map((item, i) => { item.stack = item.type; });
      }
      chart.setOption(params.newOption);
    }
  });

  var option = {
    title: {
      text: partOption.title.text,
    },
    dataZoom: [
      {
        type: 'inside'
      },
      {
        type: 'slider'
      }
    ],
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
        crossStyle: {
          color: '#999'
        }
      }
    },
    toolbox: {
      feature: {
        dataView: { show: true, readOnly: false },
        magicType: { show: true, type: ['bar', 'line'] },
        restore: { show: true },
        saveAsImage: { show: false }
      }
    },
    legend: {
      right: '10%',
      data: ['I', 'P', 'B', 'PKT'],
      selected: {
        'I': true,
        'P': true,
        'B': true,
        'PKT': false
      }
    },
    xAxis: {
      name: partOption.xAxis.name,
      type: 'category',
      axisTick: {
        alignWithLabel: true
      },
      data: partOption.xAxis.data
    },
    yAxis: {
      name: partOption.yAxis.name,
      type: 'value',
    },
    series: [
      {
        name: 'I',
        type: 'bar',
        stack: 'bar',
        tooltip: {
          valueFormatter: function (value) {
            return value == NaN ? '' : value + ' byte';
          }
        },
        large: true,
        barWidth: '30%',
        data: partOption.series[0].data
      },
      {
        name: 'P',
        type: 'bar',
        stack: 'bar',
        tooltip: {
          valueFormatter: function (value) {
            return value == NaN ? '' : value + ' byte';
          }
        },
        large: true,
        barWidth: '30%',
        data: partOption.series[1].data
      },
      {
        name: 'B',
        type: 'bar',
        stack: 'bar',
        tooltip: {
          valueFormatter: function (value) {
            return value == NaN ? '' : value + ' byte';
          }
        },
        large: true,
        barWidth: '30%',
        data: partOption.series[2].data
      },
      {
        name: 'PKT',
        type: 'bar',
        stack: 'bar',
        tooltip: {
          valueFormatter: function (value) {
            return value == NaN ? '' : value + ' byte';
          }
        },
        large: true,
        barWidth: '30%',
        data: partOption.series[3].data
      }
    ]
  };
  return option;
}
