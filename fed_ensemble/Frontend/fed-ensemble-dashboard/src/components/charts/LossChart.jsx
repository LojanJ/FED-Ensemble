/* eslint-disable react/prop-types */
import { Box, Card, CardContent } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const LossChart = ({ processedData }) => {
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
                tick={{ fontSize: 12 }}
                label={{ value: 'Loss Value', angle: -90, position: 'insideLeft', offset: 0, fontSize: 14 }}
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: "rgba(35, 35, 35, 0.9)",
                  color: "#fff", 
                  borderRadius: "8px", 
                  border: "none",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.15)"
                }}
                formatter={(value, name) => [value.toFixed(3), name]}
                labelFormatter={(label) => `Server Round: ${label}`}
              />
              <Legend 
                wrapperStyle={{ fontSize: 12, paddingTop: 20 }}
                iconSize={10}
                iconType="circle"
              />
              <defs>
                <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f44336" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#f44336" stopOpacity={0.2}/>
                </linearGradient>
              </defs>
              <Line 
                type="monotone" 
                dataKey="loss" 
                stroke="#f44336"
                strokeWidth={3}
                fill="url(#lossGradient)"
                fillOpacity={0.2}
                dot={{ r: 0 }}
                activeDot={{ r: 6, strokeWidth: 1, stroke: "#fff" }}
                name="Loss"
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
};

export default LossChart; 