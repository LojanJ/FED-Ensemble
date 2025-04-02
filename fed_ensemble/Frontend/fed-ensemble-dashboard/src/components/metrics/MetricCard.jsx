/* eslint-disable react/prop-types */
import { Card, CardContent, Box, Typography, LinearProgress } from '@mui/material';

const MetricCard = ({ 
  title, 
  value, 
  icon, 
  color, 
  description, 
  showProgress = false,
  fontWeight = '600',
  cardFontWeight = '700'
}) => {
  
  // Helper function to format values
  const formatValue = () => {
    // If value is already N/A or a string, return it as is
    if (value === 'N/A' || typeof value === 'string' && !value.match(/^[0-9.]+$/)) {
      return value;
    }
    
    // Handle percentage for Simulated Poisoning Ratio
    if (title === "Simulated Poisoning Ratio") {
      // If value is numeric or convertible to number, format as percentage
      const numVal = parseFloat(value);
      if (!isNaN(numVal)) {
        return `${(numVal * 100).toFixed(1)}%`;
      }
      return value; // Return as is if not convertible
    }
    
    return value;
  };
  
  // Determine the text color based on the metrics
  const getTextColor = () => {
    if (title === "Simulated Poisoning Ratio") {
      return "error.main";
    } else if (title === "Detection Accuracy") {
      const accuracy = parseFloat(value);
      if (accuracy > 0.75) return "success.main";
      if (accuracy > 0.5) return "warning.main";
      return "error.main";
    }
    return "text.primary";
  };
  
  return (
    <Card 
      sx={{ 
        height: '100%',
        transition: 'all 0.3s ease',
        borderLeft: `4px solid ${color}.main`,
        borderRadius: '8px',
        boxShadow: title === "Detection Accuracy" || title === "Simulated Poisoning Ratio" ? 3 : 2,
        '&:hover': {
          transform: 'translateY(-3px)',
          boxShadow: title === "Detection Accuracy" || title === "Simulated Poisoning Ratio" ? 5 : 4,
        },
      }}
    >
      <CardContent sx={{ height: '100%', p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 1.5 }}>
          <Typography variant="subtitle1" fontWeight={fontWeight} color="text.primary" sx={{ lineHeight: 1.2 }}>
            {title}
          </Typography>
          <Box 
            sx={{ 
              p: 0.7, 
              borderRadius: 1, 
              bgcolor: `${color}.main`, 
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: 0.85
            }}
          >
            {icon}
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'baseline', mt: 0.5 }}>
          <Typography 
            variant="h5" 
            component="div" 
            fontWeight={cardFontWeight}
            color={getTextColor()}
          >
            {formatValue()}
          </Typography>
        </Box>
        
        {showProgress && title === "Detection Accuracy" && (
          <Box sx={{ mt: 1.2, width: '100%', height: 3 }}>
            <LinearProgress 
              variant="determinate" 
              value={parseFloat(value) * 100}
              color={parseFloat(value) > 0.75 ? "success" : parseFloat(value) > 0.5 ? "warning" : "error"}
              sx={{ height: 3, borderRadius: 1 }}
            />
          </Box>
        )}
        
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', opacity: 0.8 }}>
          {description}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default MetricCard; 