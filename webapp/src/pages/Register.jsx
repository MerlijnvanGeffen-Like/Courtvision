import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import './Page.css'
import './Login.css'

function Register() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { register } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Wachtwoorden komen niet overeen')
      return
    }

    if (password.length < 6) {
      setError('Wachtwoord moet minimaal 6 tekens lang zijn')
      return
    }

    setLoading(true)
    const result = await register(username, email, password)
    setLoading(false)

    if (result.success) {
      navigate('/')
    } else {
      setError(result.error || 'Registratie mislukt')
    }
  }

  return (
    <div className="auth-page register-page">
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
            <Link to="/login" className="auth-tab">
              Login
            </Link>
            <Link to="/register" className="auth-tab auth-tab-active">
              Sign Up
            </Link>
          </div>

          {error && <div className="auth-error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <div className="input-wrapper">
                <div className="input-icon">
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M10 10C12.7614 10 15 7.76142 15 5C15 2.23858 12.7614 0 10 0C7.23858 0 5 2.23858 5 5C5 7.76142 7.23858 10 10 10Z" stroke="#99a1af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M10 12.5C6.66667 12.5 0 14.175 0 17.5V20H20V17.5C20 14.175 13.3333 12.5 10 12.5Z" stroke="#99a1af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <input
                  type="text"
                  id="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  disabled={loading}
                  placeholder="Kies een gebruikersnaam"
                />
              </div>
            </div>

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
                  minLength={6}
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

            <div className="form-group">
              <label htmlFor="confirmPassword">Bevestig Wachtwoord</label>
              <div className="input-wrapper">
                <div className="input-icon">
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M15.8333 9.16667H4.16667C3.24619 9.16667 2.5 9.91286 2.5 10.8333V15.8333C2.5 16.7538 3.24619 17.5 4.16667 17.5H15.8333C16.7538 17.5 17.5 16.7538 17.5 15.8333V10.8333C17.5 9.91286 16.7538 9.16667 15.8333 9.16667Z" stroke="#99a1af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M5.83333 9.16667V5.83333C5.83333 4.72827 6.27232 3.66846 7.05372 2.88706C7.83512 2.10565 8.89493 1.66667 9.99999 1.66667C11.1051 1.66667 12.1649 2.10565 12.9463 2.88706C13.7277 3.66846 14.1667 4.72827 14.1667 5.83333V9.16667" stroke="#99a1af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  id="confirmPassword"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  disabled={loading}
                  placeholder="••••••••"
                  minLength={6}
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  aria-label={showConfirmPassword ? "Wachtwoord verbergen" : "Wachtwoord tonen"}
                >
                  <img src="/icons/password.svg" alt={showConfirmPassword ? "Hide password" : "Show password"} className="password-toggle-icon" />
                </button>
              </div>
            </div>

            <button type="submit" className="auth-submit-button" disabled={loading}>
              <img src="/icons/createaccount.svg" alt="Account aanmaken" className="auth-button-icon" />
              <span>{loading ? 'Account aanmaken...' : 'Account aanmaken'}</span>
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

export default Register
