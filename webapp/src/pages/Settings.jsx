import { useState } from 'react'
import { useTheme } from '../context/ThemeContext'
import Header from '../components/Header'
import BottomNav from '../components/BottomNav'
import './Page.css'
import './Settings.css'

function Settings() {
  const { theme, toggleTheme } = useTheme()
  const [sensitivity, setSensitivity] = useState(50) // 0-100
  const [deviceStatus, setDeviceStatus] = useState('Connected')

  const sensitivityLabel = sensitivity < 33 ? 'Low' : sensitivity < 67 ? 'Medium' : 'High'
  const themeLabel = theme === 'dark' ? 'Dark' : 'Light'

  return (
    <div className="page settings-page">
      <Header />
      
      <div className="page-content">
        <div className="settings-container">
          <h2 className="settings-title">Settings</h2>
          
          <div className="settings-sections">
            <div className="setting-section">
              <h3 className="setting-section-title">Appearance</h3>
              <p className="setting-section-description">Choose your preferred color theme</p>
              <div className="setting-row">
                <span className="setting-label">Theme</span>
                <button className="theme-button" onClick={toggleTheme}>
                  {theme === 'dark' ? (
                    <img src="/icons/dark.svg" alt="Dark" className="theme-icon" />
                  ) : (
                    <img src="/icons/light.svg" alt="Light" className="theme-icon" />
                  )}
                  {themeLabel}
                </button>
              </div>
            </div>

            <div className="setting-section">
              <h3 className="setting-section-title">Camera Settings</h3>
              <p className="setting-section-description">Configure camera detection sensitivity and position</p>
              <div className="sensitivity-control">
                <div className="sensitivity-header">
                  <span className="setting-label">Sensitivity</span>
                  <span className="sensitivity-value">{sensitivityLabel}</span>
                </div>
                <div className="slider-container" style={{ '--slider-value': `${sensitivity}%` }}>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={sensitivity}
                    onChange={(e) => setSensitivity(Number(e.target.value))}
                    className="sensitivity-slider"
                  />
                </div>
              </div>
            </div>

            <div className="setting-section">
              <h3 className="setting-section-title">Hardware Device</h3>
              <p className="setting-section-description">
                Status: <span className={`status-text ${deviceStatus === 'Connected' ? 'connected' : 'disconnected'}`}>
                  {deviceStatus}
                </span>
              </p>
              <p className="setting-section-description">Device placed at the sideline</p>
            </div>

            <div className="setting-section">
              <h3 className="setting-section-title">About</h3>
              <p className="setting-section-description">Courtvision v1.0.0</p>
              <p className="setting-section-description" style={{ fontSize: '12px', marginTop: '8px' }}>
                Smart basketball scoring system with computer vision
              </p>
            </div>
          </div>
        </div>
      </div>

      <BottomNav />
    </div>
  )
}

export default Settings

