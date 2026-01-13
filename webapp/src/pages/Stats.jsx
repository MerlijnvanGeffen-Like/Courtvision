import { useState, useEffect } from 'react'
import Header from '../components/Header'
import BottomNav from '../components/BottomNav'
import { statsAPI } from '../utils/api'
import './Page.css'
import './Stats.css'

function Stats() {
  const [stats, setStats] = useState({
    current_session: {
      shots_made: 0,
      shots_missed: 0,
      total_shots: 0,
      accuracy: 0.0
    },
    all_time: {
      total_sessions: 0,
      total_shots_made: 0,
      total_shots_missed: 0,
      total_shots: 0,
      average_accuracy: 0.0,
      total_play_time_seconds: 0,
      total_play_time_formatted: '0m'
    }
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await statsAPI.getStats()
        setStats(data)
      } catch (error) {
        console.error('Error fetching stats:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
    // Refresh stats every 5 seconds
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [])

  const currentMade = stats.current_session?.shots_made || 0
  const currentAccuracy = stats.current_session?.accuracy || 0.0
  const totalSessions = stats.all_time?.total_sessions || 0
  const totalShotsMade = stats.all_time?.total_shots_made || 0
  const averageAccuracy = stats.all_time?.average_accuracy || 0.0
  const totalPlayTime = stats.all_time?.total_play_time_formatted || '0m'

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

