/* eslint-disable react/prop-types */
import { Box, Card, CardContent } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const AccuracyChart = ({ processedData }) => {
  return (
    <Card sx={{ flex: 1 }}>
      <CardContent>
        <Box sx={{ height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(150, 150, 150, 0.1)" />
              <XAxis 
                dataKey="round" 
                tick={{ fontSize: 12 }} 
                tickCount={10}
                label={{ value: 'Server Round', position: 'insideBottom', offset: -5, fontSize: 14 }}
              />
              <YAxis 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                domain={[0, 1]}
                tick={{ fontSize: 12 }}
                label={{ value: 'Metric Value', angle: -90, position: 'insideLeft', offset: 0, fontSize: 14 }}
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: "rgba(35, 35, 35, 0.9)",
                  color: "#fff", 
                  borderRadius: "8px", 
                  border: "none",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.15)"
                }}
                formatter={(value, name) => [(value * 100).toFixed(2) + "%", name]}
                labelFormatter={(label) => `Server Round: ${label}`}
              />
              <Legend 
                wrapperStyle={{ fontSize: 12, paddingTop: 20 }}
                iconSize={10}
                iconType="circle"
              />
              <defs>
                <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3F88E3" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3F88E3" stopOpacity={0.2}/>
                </linearGradient>
                <linearGradient id="precisionGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4caf50" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#4caf50" stopOpacity={0.2}/>
                </linearGradient>
              </defs>
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#3F88E3" 
                strokeWidth={3}
                dot={{ r: 0 }}
                activeDot={{ r: 6, strokeWidth: 1, stroke: "#fff" }}
                name="Accuracy"
                connectNulls
              />
              <Line 
                type="monotone" 
                dataKey="precision" 
                stroke="#4caf50" 
                strokeWidth={2}
                dot={{ r: 0 }}
                activeDot={{ r: 4 }}
                strokeDasharray="5 5" 
                name="Precision"
                connectNulls
              />
              <Line 
                type="monotone" 
                dataKey="recall" 
                stroke="#ff9800" 
                strokeWidth={2}
                dot={{ r: 0 }}
                activeDot={{ r: 4 }}
                strokeDasharray="3 3" 
                name="Recall"
                connectNulls
              />
              <Line 
                type="monotone" 
                dataKey="f1" 
                stroke="#9c27b0" 
                strokeWidth={2}
                dot={{ r: 0 }}
                activeDot={{ r: 4 }}
                strokeDasharray="7 7" 
                name="F1"
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
};

export default AccuracyChart; 