/* eslint-disable react/prop-types */
import Dashboard from './components/dashboard/Dashboard';
import { AppThemeProvider } from './context/ThemeContext';

const App = () => {
  return (
    <AppThemeProvider>
      <Dashboard />
    </AppThemeProvider>
  );
};

export default App;