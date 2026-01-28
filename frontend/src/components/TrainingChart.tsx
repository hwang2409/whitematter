import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface DataPoint {
  epoch: number;
  loss: number;
  accuracy: number;
}

interface TrainingChartProps {
  data: DataPoint[];
}

export default function TrainingChart({ data }: TrainingChartProps) {
  if (data.length === 0) {
    return null;
  }

  return (
    <div className="training-chart">
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="epoch"
            stroke="#666"
            tick={{ fill: '#999' }}
            label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#999' }}
          />
          <YAxis
            yAxisId="left"
            stroke="#ff6b6b"
            tick={{ fill: '#ff6b6b' }}
            label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#ff6b6b' }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#4ecdc4"
            tick={{ fill: '#4ecdc4' }}
            domain={[0, 100]}
            label={{ value: 'Accuracy %', angle: 90, position: 'insideRight', fill: '#4ecdc4' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#fff' }}
          />
          <Legend wrapperStyle={{ paddingTop: '10px' }} />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="loss"
            stroke="#ff6b6b"
            strokeWidth={2}
            dot={{ fill: '#ff6b6b', r: 3 }}
            activeDot={{ r: 5 }}
            name="Loss"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="accuracy"
            stroke="#4ecdc4"
            strokeWidth={2}
            dot={{ fill: '#4ecdc4', r: 3 }}
            activeDot={{ r: 5 }}
            name="Accuracy"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
