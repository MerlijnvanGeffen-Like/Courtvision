import { useState } from 'react'
import Header from '../components/Header'
import BottomNav from '../components/BottomNav'
import './Page.css'
import './Home.css'

function Home() {
  const [score, setScore] = useState(0)
  const [madeShots, setMadeShots] = useState(0)
  const [totalShots, setTotalShots] = useState(0)
  const [time, setTime] = useState('00:00')
  const [isActive, setIsActive] = useState(false)
  const [isPaused, setIsPaused] = useState(true)
  const [cameraActive, setCameraActive] = useState(false)

  const accuracy = totalShots > 0 ? ((madeShots / totalShots) * 100).toFixed(1) : 0

  const handleStart = () => {
    if (!cameraActive) {
      setCameraActive(true)
    } else {
      setIsPaused(!isPaused)
      setIsActive(!isPaused)
    }
  }

  const handleStopCamera = () => {
    setCameraActive(false)
    setIsActive(false)
    setIsPaused(true)
  }

  return (
    <div className="page home-page">
      <Header />
      
      <div className="page-content">
        <div className="score-display">
          <div className="score-card">
            <div className="score-header">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M8.33337 12.2167V13.5717C8.33023 13.8572 8.25378 14.1371 8.11136 14.3846C7.96894 14.6321 7.76532 14.8388 7.52004 14.985C6.99933 15.3707 6.57574 15.8725 6.28294 16.4506C5.99015 17.0286 5.83622 17.667 5.83337 18.315" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M11.6666 12.2167V13.5717C11.6698 13.8572 11.7462 14.1371 11.8886 14.3846C12.0311 14.6321 12.2347 14.8388 12.48 14.985C13.0007 15.3707 13.4243 15.8725 13.7171 16.4506C14.0098 17.0286 14.1638 17.667 14.1666 18.315" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M15 7.50001H16.25C16.8025 7.50001 17.3324 7.28052 17.7231 6.88982C18.1138 6.49912 18.3333 5.96921 18.3333 5.41668C18.3333 4.86414 18.1138 4.33424 17.7231 3.94354C17.3324 3.55284 16.8025 3.33334 16.25 3.33334H15" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M3.33337 18.3333H16.6667" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M5 7.49999C5 8.82607 5.52678 10.0978 6.46447 11.0355C7.40215 11.9732 8.67392 12.5 10 12.5C11.3261 12.5 12.5979 11.9732 13.5355 11.0355C14.4732 10.0978 15 8.82607 15 7.49999V2.49999C15 2.27898 14.9122 2.06701 14.7559 1.91073C14.5996 1.75445 14.3877 1.66666 14.1667 1.66666H5.83333C5.61232 1.66666 5.40036 1.75445 5.24408 1.91073C5.0878 2.06701 5 2.27898 5 2.49999V7.49999Z" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M4.99996 7.50001H3.74996C3.19742 7.50001 2.66752 7.28052 2.27682 6.88982C1.88612 6.49912 1.66663 5.96921 1.66663 5.41668C1.66663 4.86414 1.88612 4.33424 2.27682 3.94354C2.66752 3.55284 3.19742 3.33334 3.74996 3.33334H4.99996" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span>Score</span>
            </div>
            <div className="score-value">{score}</div>
            <div className="score-label">Made Shots</div>
          </div>

          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-header">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <g clipPath="url(#clip0_1_43)">
                    <path d="M8.00004 14.6666C11.6819 14.6666 14.6667 11.6819 14.6667 7.99998C14.6667 4.31808 11.6819 1.33331 8.00004 1.33331C4.31814 1.33331 1.33337 4.31808 1.33337 7.99998C1.33337 11.6819 4.31814 14.6666 8.00004 14.6666Z" stroke="currentColor" strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M8 12C10.2091 12 12 10.2091 12 8C12 5.79086 10.2091 4 8 4C5.79086 4 4 5.79086 4 8C4 10.2091 5.79086 12 8 12Z" stroke="currentColor" strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M7.99996 9.33335C8.73634 9.33335 9.33329 8.7364 9.33329 8.00002C9.33329 7.26364 8.73634 6.66669 7.99996 6.66669C7.26358 6.66669 6.66663 7.26364 6.66663 8.00002C6.66663 8.7364 7.26358 9.33335 7.99996 9.33335Z" stroke="currentColor" strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
                  </g>
                  <defs>
                    <clipPath id="clip0_1_43">
                      <rect width="16" height="16" fill="white"/>
                    </clipPath>
                  </defs>
                </svg>
                <span>Accuracy</span>
              </div>
              <div className="stat-value accuracy">{accuracy}%</div>
              <div className="stat-subvalue">{madeShots}/{totalShots}</div>
            </div>

            <div className="stat-card">
              <div className="stat-header">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M6.66663 1.33331H9.33329" stroke="currentColor" strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M8 9.33331L10 7.33331" stroke="currentColor" strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M7.99996 14.6667C10.9455 14.6667 13.3333 12.2789 13.3333 9.33333C13.3333 6.38781 10.9455 4 7.99996 4C5.05444 4 2.66663 6.38781 2.66663 9.33333C2.66663 12.2789 5.05444 14.6667 7.99996 14.6667Z" stroke="currentColor" strokeWidth="1.33333" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span>Time</span>
              </div>
              <div className="stat-value time">{time}</div>
              <div className="stat-subvalue">{isPaused ? 'Paused' : 'Active'}</div>
            </div>
          </div>
        </div>

        <div className="camera-view">
          {cameraActive ? (
            <>
              <div className="camera-overlay">
                <div className="live-indicator">
                  <div className="live-dot"></div>
                  <span>LIVE</span>
                </div>
              </div>
              <button className="stop-camera-btn" onClick={handleStopCamera}>
                <img src="/icons/stopcam.svg" alt="Stop Camera" className="icon-img" />
                Stop Camera
              </button>
            </>
          ) : (
            <div className="camera-inactive">
              <div className="camera-icon">
                <img src="/icons/camerainactive.svg" alt="Camera Inactive" className="camera-icon-img" />
              </div>
              <h3>Camera Inactive</h3>
              <p>Start camera to begin tracking</p>
              <button className="start-camera-btn" onClick={handleStart}>
                <img src="/icons/startcam.svg" alt="Start Camera" className="icon-img" />
                Start Camera
              </button>
            </div>
          )}
        </div>

        <div className="controls">
          <button 
            className="primary-btn" 
            onClick={handleStart}
          >
            {isPaused ? (
              <img src="/icons/play.svg" alt="Play" className="icon-img" />
            ) : (
              <img src="/icons/pause.svg" alt="Pause" className="icon-img" />
            )}
            {isPaused ? 'Start' : 'Pause'}
          </button>
          <button className="secondary-btn">
            <img src="/icons/restart.svg" alt="Restart" className="icon-img" />
          </button>
        </div>
      </div>

      <BottomNav />
    </div>
  )
}

export default Home

