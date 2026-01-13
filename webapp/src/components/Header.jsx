import './Header.css'

function Header() {

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
      </div>
    </div>
  )
}

export default Header

