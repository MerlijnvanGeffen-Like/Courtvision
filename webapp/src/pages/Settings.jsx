import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'
import { useAuth } from '../context/AuthContext'
import { healthAPI } from '../utils/api'
import Header from '../components/Header'
import BottomNav from '../components/BottomNav'
import './Page.css'
import './Settings.css'

function Settings() {
  const { theme, toggleTheme } = useTheme()
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [deviceStatus, setDeviceStatus] = useState('Checking...')

  // Check backend connection status
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        await healthAPI.check()
        setDeviceStatus('Connected')
      } catch (error) {
        setDeviceStatus('Disconnected')
      }
    }

    // Check immediately
    checkBackendStatus()

    // Check every 5 seconds
    const interval = setInterval(checkBackendStatus, 5000)

    return () => clearInterval(interval)
  }, [])

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

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
              <h3 className="setting-section-title">User Info</h3>
              <p className="setting-section-description">Manage your account information</p>
              {user && (
                <div className="user-info-content">
                  <div className="user-info-row">
                    <span className="setting-label">Username</span>
                    <span className="user-username">{user.username}</span>
                  </div>
                  {user.email && (
                    <div className="user-info-row">
                      <span className="setting-label">Email</span>
                      <span className="user-email">{user.email}</span>
                    </div>
                  )}
                  <button className="user-logout-button" onClick={handleLogout}>
                    Logout
                  </button>
                </div>
              )}
            </div>

            <div className="setting-section">
              <h3 className="setting-section-title">Hardware Device</h3>
              <p className="setting-section-description">
                Status: <span className={`status-text ${deviceStatus === 'Connected' ? 'connected' : deviceStatus === 'Disconnected' ? 'disconnected' : 'checking'}`}>
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

