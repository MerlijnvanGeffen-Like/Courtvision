import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Header.css'

function Header() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="header">
      <div className="header-content">
        <div className="logo-container">
          <div className="logo">
            <img src="/images/logo.png" alt="Courtvision Logo" className="logo-image" />
          </div>
          <div className="logo-text">
            <h1 className="logo-title">Courtvision</h1>
            <p className="logo-subtitle">Basketball Tracking</p>
          </div>
        </div>
        {user && (
          <div className="header-user">
            <span className="header-username">{user.username}</span>
            <button className="header-logout" onClick={handleLogout}>
              Logout
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default Header

