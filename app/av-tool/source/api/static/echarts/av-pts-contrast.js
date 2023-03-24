function getAVPtsContrastOption(partOption) {
  const colors = ['#5470C6', '#EE6666'];
  const legends = [partOption.series[0].name, partOption.series[1].name];
  if (partOption.xAxis.length < 2)
    partOption.xAxis.push({name: '', show: false, data: []});
  if (partOption.yAxis.length < 2)
    partOption.yAxis.push({name: '', show: false});
  var option = {
    title: {
      text: partOption.title.text,
    },
    color: colors,
    dataZoom: [
      {
        type: 'inside'
      },
      {
        type: 'slider'
      }
    ],
    tooltip: {
      trigger: 'none',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      right: '10%'
    },
    grid: {
      top: 60,
      left: '8%',
      right: '8%'
    },
    xAxis: [
      {
        name: partOption.xAxis[0].name,
        type: 'category',
        position: 'bottom',
        axisTick: {
          alignWithLabel: true
        },
        axisLine: {
          onZero: false,
          lineStyle: {
            color: colors[0]
          }
        },
        axisPointer: {
          label: {
            formatter: function (params) {
              return (
                legends[0] + ' ' + params.value +
                (params.seriesData.length ? '：' + params.seriesData[0].data : '')
              );
            }
          }
        },
        data: partOption.xAxis[0].data
      },
      {
        name: partOption.xAxis[1].name,
        show: partOption.xAxis[1].show,
        type: 'category',
        position: 'top',
        axisTick: {
          alignWithLabel: true
        },
        axisLine: {
          onZero: false,
          lineStyle: {
            color: colors[1]
          }
        },
        axisPointer: {
          label: {
            formatter: function (params) {
              return (
                legends[1] + ' ' + params.value +
                (params.seriesData.length ? '：' + params.seriesData[0].data : '')
              );
            }
          }
        },
        data: partOption.xAxis[0].data
      }
    ],
    yAxis: [
      {
        name: partOption.yAxis[0].name,
        type: 'value',
        position: 'left',
        axisLine: {
          onZero: false,
          lineStyle: {
            color: colors[0]
          }
        },
        axisLabel: {
          formatter: '{value}'
        },
        splitLine: {
          show: true
        }
      },
      {
        name: partOption.yAxis[1].name,
        type: 'value',
        show: partOption.yAxis[1].show,
        position: 'right',
        axisLine: {
          onZero: false,
          lineStyle: {
            color: colors[1]
          }
        },
        axisLabel: {
          formatter: '{value}'
        },
        splitLine: {
          show: true
        }
      }
    ],
    series: [
      {
        name: legends[0],
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        smooth: true,
        emphasis: {
          focus: 'series'
        },
        data: partOption.series[0].data
      },
      {
        name: legends[1],
        type: 'line',
        smooth: true,
        xAxisIndex: 1,
        yAxisIndex: 0,
        emphasis: {
          focus: 'series'
        },
        data: partOption.series[1].data
      }
    ]
  };
  return option;
}
