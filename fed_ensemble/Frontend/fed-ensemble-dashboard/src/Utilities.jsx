import { useState, useEffect } from 'react';
import { 
    FormControlLabel,
    Switch
} from '@mui/material';
import { 
    DarkMode as DarkModeIcon,
    LightMode as LightModeIcon  
} from '@mui/icons-material';

function DarkModeToggle() {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("theme") === "dark"
  );

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  return(
    <FormControlLabel
      control={
        <Switch
          checked={darkMode}
          onChange={() => setDarkMode(!darkMode)}
          color='default'
          />
        }
      label = {darkMode ? <DarkModeIcon/> : <LightModeIcon/>}
    />
  )
}

export default DarkModeToggle;