import Header from '../components/Header'
import BottomNav from '../components/BottomNav'
import './Page.css'
import './Leaderboard.css'

function Leaderboard() {
  const players = [
    { rank: 1, name: 'Micheal J.', score: 2654, accuracy: 87, isTop3: true },
    { rank: 2, name: 'LeBron J.', score: 2522, accuracy: 82, isTop3: true },
    { rank: 3, name: 'Kobe B.', score: 2371, accuracy: 84, isTop3: true },
    { rank: 4, name: 'Steph C.', score: 2215, accuracy: 91, isTop3: false },
    { rank: 5, name: 'Kevin D.', score: 2089, accuracy: 78, isTop3: false },
    { rank: 6, name: 'O\'Neal S.', score: 1842, accuracy: 72, isTop3: false },
  ]

  const userRank = 156
  const topScore = 2847
  const totalPlayers = 1234

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
              <div className="summary-value">#{userRank}</div>
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
            {players.map((player) => (
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
                    <h3 className="player-name">{player.name}</h3>
                    {player.isTop3 && (
                      <span className="top3-badge">Top 3</span>
                    )}
                  </div>
                  <div className="player-details">
                    {player.sessions ? (
                      <>
                        <span>{player.sessions} sessies</span>
                        <span>•</span>
                        <span>{player.accuracy}% accuraatheid</span>
                      </>
                    ) : (
                      <span>{player.accuracy}% accuracy</span>
                    )}
                  </div>
                </div>
                <div className="player-score">
                  <div className="player-score-value">{player.score.toLocaleString()}</div>
                  <div className="player-score-label">scored</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <BottomNav />
    </div>
  )
}

export default Leaderboard

