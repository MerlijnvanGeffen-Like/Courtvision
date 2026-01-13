import { useState, useEffect } from 'react'
import Header from '../components/Header'
import BottomNav from '../components/BottomNav'
import { leaderboardAPI } from '../utils/api'
import './Page.css'
import './Leaderboard.css'

function Leaderboard() {
  const [leaderboard, setLeaderboard] = useState({
    players: [],
    user_rank: null,
    top_score: 0,
    total_players: 0
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchLeaderboard = async () => {
      try {
        const data = await leaderboardAPI.getLeaderboard()
        setLeaderboard(data)
      } catch (error) {
        console.error('Error fetching leaderboard:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchLeaderboard()
    // Refresh leaderboard every 30 seconds
    const interval = setInterval(fetchLeaderboard, 30000)
    return () => clearInterval(interval)
  }, [])

  const players = leaderboard.players
  const userRank = leaderboard.user_rank
  const topScore = leaderboard.top_score
  const totalPlayers = leaderboard.total_players

  return (
    <div className="page leaderboard-page">
      <Header />
      
      <div className="page-content">
        <div className="leaderboard-container">
          <div className="leaderboard-header">
            <div>
              <h2 className="leaderboard-title">Leaderboard</h2>
              <p className="leaderboard-subtitle">Top 10 players</p>
            </div>
          </div>

          <div className="summary-cards">
            <div className="summary-card">
              <img src="/icons/yourrank.svg" alt="Your Rank" className="summary-card-icon" />
              <div className="summary-label">Your Rank</div>
              <div className="summary-value">
                {userRank ? `#${userRank}` : 'N/A'}
              </div>
            </div>
            <div className="summary-card">
              <img src="/icons/topscore.svg" alt="Top Score" className="summary-card-icon" />
              <div className="summary-label">Top Score</div>
              <div className="summary-value" style={{ color: '#ff8c42' }}>{topScore}</div>
            </div>
            <div className="summary-card">
              <img src="/icons/players.svg" alt="Players" className="summary-card-icon" />
              <div className="summary-label">Players</div>
              <div className="summary-value" style={{ color: '#1ec5ff' }}>{totalPlayers.toLocaleString()}</div>
            </div>
          </div>

          <div className="players-list">
            {loading ? (
              <div style={{ padding: '2rem', textAlign: 'center' }}>Loading...</div>
            ) : players.length === 0 ? (
              <div style={{ padding: '2rem', textAlign: 'center' }}>No players yet. Be the first!</div>
            ) : (
              players.map((player) => (
              <div 
                key={player.rank} 
                className={`player-card ${player.isTop3 ? 'top3' : ''}`}
              >
                {player.isTop3 ? (
                  <div className="player-medal">
                    <img src="/icons/top3.svg" alt="Top 3" className="player-medal-icon" />
                  </div>
                ) : (
                  <div className="player-rank">#{player.rank}</div>
                )}
                <div className="player-info">
                  <div className="player-header">
                    <h3 className="player-name">{player.username}</h3>
                    {player.isTop3 && (
                      <span className="top3-badge">Top 3</span>
                    )}
                  </div>
                  <div className="player-details">
                    <span>{player.average_accuracy}% accuracy</span>
                  </div>
                </div>
                <div className="player-score">
                  <div className="player-score-value">{player.total_shots_made.toLocaleString()}</div>
                  <div className="player-score-label">scored</div>
                </div>
              </div>
              ))
            )}
          </div>
        </div>
      </div>

      <BottomNav />
    </div>
  )
}

export default Leaderboard

