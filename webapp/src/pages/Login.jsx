import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Page.css'
import './Login.css'

function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    // API supports both email and username for login
    const result = await login(email, password)
    setLoading(false)

    if (result.success) {
      navigate('/')
    } else {
      setError(result.error || 'Login failed')
    }
  }

  return (
    <div className="auth-page login-page">
      <div className="auth-wrapper">
        <div className="auth-header-section">
          <div className="auth-logo-container">
            <div className="auth-logo-wrapper">
              <div className="auth-logo-image">
                <img src="/images/logo.png" alt="Courtvision Logo" />
              </div>
            </div>
            <h1 className="auth-brand-title">Courtvision</h1>
            <p className="auth-brand-subtitle">Smart Basketball Tracking System</p>
          </div>
        </div>

        <div className="auth-card">
          <div className="auth-tabs">
            <Link to="/login" className="auth-tab auth-tab-active">
              Login
            </Link>
            <Link to="/register" className="auth-tab">
              Sign Up
            </Link>
          </div>

          {error && <div className="auth-error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <div className="input-wrapper">
                <div className="input-icon">
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M2.5 6.66667L9.0755 11.0504C9.63533 11.4236 10.3647 11.4236 10.9245 11.0504L17.5 6.66667M4.16667 15.8333H15.8333C16.7538 15.8333 17.5 15.0872 17.5 14.1667V5.83333C17.5 4.91286 16.7538 4.16667 15.8333 4.16667H4.16667C3.24619 4.16667 2.5 4.91286 2.5 5.83333V14.1667C2.5 15.0872 3.24619 15.8333 4.16667 15.8333Z" stroke="#99a1af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={loading}
                  placeholder="je@email.com"
                />
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="password">Wachtwoord</label>
              <div className="input-wrapper">
                <div className="input-icon">
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.8333 9.16667H4.16667C3.24619 9.16667 2.5 9.91286 2.5 10.8333V15.8333C2.5 16.7538 3.24619 17.5 4.16667 17.5H15.8333C16.7538 17.5 17.5 16.7538 17.5 15.8333V10.8333C17.5 9.91286 16.7538 9.16667 15.8333 9.16667Z" stroke="#99a1af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M5.83333 9.16667V5.83333C5.83333 4.72827 6.27232 3.66846 7.05372 2.88706C7.83512 2.10565 8.89493 1.66667 9.99999 1.66667C11.1051 1.66667 12.1649 2.10565 12.9463 2.88706C13.7277 3.66846 14.1667 4.72827 14.1667 5.83333V9.16667" stroke="#99a1af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <input
                  type={showPassword ? "text" : "password"}
                  id="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={loading}
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Wachtwoord verbergen" : "Wachtwoord tonen"}
                >
                  <img src="/icons/password.svg" alt={showPassword ? "Hide password" : "Show password"} className="password-toggle-icon" />
                </button>
              </div>
            </div>

            <Link to="/forgot-password" className="forgot-password-link">
              Wachtwoord vergeten?
            </Link>

            <button type="submit" className="auth-submit-button" disabled={loading}>
              <img src="/icons/inloggen.svg" alt="Inloggen" className="auth-button-icon" />
              <span>{loading ? 'Inloggen...' : 'Inloggen'}</span>
            </button>
          </form>
        </div>

        <div className="auth-footer">
          <p className="auth-footer-copyright">© 2026 Courtvision. All rights reserved.</p>
          <div className="auth-footer-links">
            <Link to="/privacy">Privacy Policy</Link>
            <span className="auth-footer-separator">•</span>
            <Link to="/terms">Terms of Service</Link>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Login
