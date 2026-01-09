import Header from '../components/Header'
import BottomNav from '../components/BottomNav'
import './Page.css'
import './Stats.css'

function Stats() {
  const currentMade = 0
  const currentAccuracy = 0.0
  const totalSessions = 0
  const totalShotsMade = 0
  const averageAccuracy = 0.0
  const totalPlayTime = '0m'

  return (
    <div className="page stats-page">
      <Header />
      
      <div className="page-content">
        <div className="stats-container">
          <div className="section">
            <h2 className="section-title">
              <img src="/icons/currentsession.svg" alt="Current Session" className="section-icon" />
              Current Session
            </h2>
            
            <div className="stats-grid">
              <div className="stat-box">
                <div className="stat-box-header">
                  <img src="/icons/made.svg" alt="Made" className="stat-box-icon" />
                  <span>Made</span>
                </div>
                <div className="stat-box-value">{currentMade}</div>
              </div>
              
              <div className="stat-box">
                <div className="stat-box-header">
                  <img src="/icons/accuracy.svg" alt="Accuracy" className="stat-box-icon" />
                  <span>Accuracy</span>
                </div>
                <div className="stat-box-value accuracy">{currentAccuracy}%</div>
              </div>
            </div>
          </div>

          <div className="section">
            <h2 className="section-title">
              <img src="/icons/alltimestats.svg" alt="All Time Stats" className="section-icon" />
              All Time Stats
            </h2>
            
            <div className="all-time-stats">
              <div className="stat-item">
                <div className="stat-item-icon" style={{ backgroundColor: 'rgba(255, 140, 66, 0.2)' }}>
                  <img src="/icons/totalsessions.svg" alt="Total Sessions" className="stat-item-icon-img" />
                </div>
                <div className="stat-item-content">
                  <div className="stat-item-label">Total Sessions</div>
                  <div className="stat-item-value">{totalSessions}</div>
                </div>
              </div>

              <div className="stat-item">
                <div className="stat-item-icon" style={{ backgroundColor: 'rgba(30, 197, 255, 0.2)' }}>
                  <img src="/icons/totalshotsmade.svg" alt="Total Shots Made" className="stat-item-icon-img" />
                </div>
                <div className="stat-item-content">
                  <div className="stat-item-label">Total Shots Made</div>
                  <div className="stat-item-value" style={{ color: '#1ec5ff' }}>{totalShotsMade}</div>
                </div>
              </div>

              <div className="stat-item">
                <div className="stat-item-icon" style={{ backgroundColor: 'rgba(255, 140, 66, 0.2)' }}>
                  <img src="/icons/averageaccuracy.svg" alt="Average Accuracy" className="stat-item-icon-img" />
                </div>
                <div className="stat-item-content">
                  <div className="stat-item-label">Average Accuracy</div>
                  <div className="stat-item-value" style={{ color: '#ff8c42' }}>{averageAccuracy}%</div>
                </div>
              </div>

              <div className="stat-item">
                <div className="stat-item-icon" style={{ backgroundColor: 'rgba(30, 197, 255, 0.2)' }}>
                  <img src="/icons/totalplaytime.svg" alt="Total Play Time" className="stat-item-icon-img" />
                </div>
                <div className="stat-item-content">
                  <div className="stat-item-label">Total Play Time</div>
                  <div className="stat-item-value" style={{ color: '#1ec5ff' }}>{totalPlayTime}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="motivation-box">
            <p>Keep shooting to improve your stats! The more you practice, the better your accuracy will become.</p>
          </div>
        </div>
      </div>

      <BottomNav />
    </div>
  )
}

export default Stats

