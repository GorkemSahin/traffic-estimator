import React, { useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';

const getDate = (date) => (
  new Date(parseInt(date)).toLocaleString('en-GB', { month: 'short', day: 'numeric', hour: 'numeric', minute: 'numeric' })
);

function Chart({ data }) {

  console.log(data)

  const [labels, setLabels] = useState();
  
  const parsedData = useMemo(() => {
    if (!data) return null;
    const validData = data.data.map(d => ({ date: getDate(d.date), size: parseFloat(d.value / (1024*1024)).toFixed(2)}))
    setLabels(validData.map(d => d.date ));
    return {
      label: 'Served Data (mb)',
      borderColor: 'gray',
      data: validData.map(d => d.size)
    }
  }, [data]);

  const parsedPredictions = useMemo(() => {
    if (!data) return null;
    let validData = data.prediction.map(d => ({ date: getDate(d.date), size: parseFloat(d.value / (1024*1024)).toFixed(2)}))
    let i;
    for (i = 0; i < data.data.length - data.prediction.length; i++) {
      validData.unshift({ date: null, size: Number.NaN })
    }
    return {
      label: 'Predicted Data (mb)',
      borderColor: '#1db954',
      data: validData.map(d => d.size)
    }
  }, [data]);

  return (
    <Line
      data={{
        labels,
        datasets: [parsedData, parsedPredictions]
      }}
      options={{
        maintainAspectRatio: false,
        spanGaps: true,
        title: {
          display: true,
          text: `Mean Absolute Percentage Error: ${data.mape}`
        }
      }}/>
  );
}

export default Chart;
