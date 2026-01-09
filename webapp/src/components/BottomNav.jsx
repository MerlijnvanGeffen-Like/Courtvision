import { Link, useLocation } from 'react-router-dom'
import './BottomNav.css'

function BottomNav() {
  const location = useLocation()

  const isActive = (path) => location.pathname === path

  return (
    <nav className="bottom-nav">
      <Link to="/" className={`nav-item ${isActive('/') ? 'active' : ''}`}>
        <div className="nav-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 21V13C15 12.7348 14.8946 12.4804 14.7071 12.2929C14.5196 12.1054 14.2652 12 14 12H10C9.73478 12 9.48043 12.1054 9.29289 12.2929C9.10536 12.4804 9 12.7348 9 13V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M3 9.99999C2.99993 9.70906 3.06333 9.42161 3.18579 9.15771C3.30824 8.8938 3.4868 8.65979 3.709 8.47199L10.709 2.47199C11.07 2.1669 11.5274 1.99951 12 1.99951C12.4726 1.99951 12.93 2.1669 13.291 2.47199L20.291 8.47199C20.5132 8.65979 20.6918 8.8938 20.8142 9.15771C20.9367 9.42161 21.0001 9.70906 21 9.99999V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V9.99999Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <span className="nav-label">Home</span>
      </Link>
      
      <Link to="/stats" className={`nav-item ${isActive('/stats') ? 'active' : ''}`}>
        <div className="nav-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 3V19C3 19.5304 3.21071 20.0391 3.58579 20.4142C3.96086 20.7893 4.46957 21 5 21H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M18 17V9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M13 17V5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M8 17V14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <span className="nav-label">Stats</span>
      </Link>
      
      <Link to="/leaderboard" className={`nav-item ${isActive('/leaderboard') ? 'active' : ''}`}>
        <div className="nav-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 14.66V16.286C9.99622 16.6287 9.90448 16.9646 9.73358 17.2615C9.56268 17.5585 9.31834 17.8066 9.024 17.982C8.39914 18.4448 7.89084 19.047 7.53948 19.7407C7.18813 20.4344 7.00341 21.2005 7 21.978" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M14 14.66V16.286C14.0038 16.6287 14.0955 16.9646 14.2664 17.2615C14.4373 17.5585 14.6817 17.8066 14.976 17.982C15.6009 18.4448 16.1092 19.047 16.4605 19.7407C16.8119 20.4344 16.9966 21.2005 17 21.978" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M18 9H19.5C20.163 9 20.7989 8.73661 21.2678 8.26777C21.7366 7.79893 22 7.16304 22 6.5C22 5.83696 21.7366 5.20107 21.2678 4.73223C20.7989 4.26339 20.163 4 19.5 4H18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M4 22H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M6 9C6 10.5913 6.63214 12.1174 7.75736 13.2426C8.88258 14.3679 10.4087 15 12 15C13.5913 15 15.1174 14.3679 16.2426 13.2426C17.3679 12.1174 18 10.5913 18 9V3C18 2.73478 17.8946 2.48043 17.7071 2.29289C17.5196 2.10536 17.2652 2 17 2H7C6.73478 2 6.48043 2.10536 6.29289 2.29289C6.10536 2.48043 6 2.73478 6 3V9Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M6 9H4.5C3.83696 9 3.20107 8.73661 2.73223 8.26777C2.26339 7.79893 2 7.16304 2 6.5C2 5.83696 2.26339 5.20107 2.73223 4.73223C3.20107 4.26339 3.83696 4 4.5 4H6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <span className="nav-label">Leaderboard</span>
      </Link>
      
      <Link to="/settings" className={`nav-item ${isActive('/settings') ? 'active' : ''}`}>
        <div className="nav-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9.67106 4.13603C9.72616 3.55637 9.99539 3.01807 10.4262 2.62631C10.8569 2.23454 11.4183 2.01746 12.0006 2.01746C12.5828 2.01746 13.1442 2.23454 13.575 2.62631C14.0057 3.01807 14.275 3.55637 14.3301 4.13603C14.3632 4.51048 14.486 4.87145 14.6882 5.18837C14.8904 5.50529 15.1659 5.76884 15.4915 5.95671C15.8171 6.14457 16.1832 6.25123 16.5588 6.26765C16.9343 6.28407 17.3083 6.20977 17.6491 6.05103C18.1782 5.81081 18.7777 5.77605 19.3311 5.95352C19.8844 6.13098 20.3519 6.50798 20.6426 7.01113C20.9333 7.51429 21.0263 8.1076 20.9037 8.6756C20.7811 9.2436 20.4515 9.74565 19.9791 10.084C19.6714 10.2999 19.4203 10.5866 19.247 10.9201C19.0736 11.2535 18.9831 11.6237 18.9831 11.9995C18.9831 12.3753 19.0736 12.7456 19.247 13.079C19.4203 13.4124 19.6714 13.6992 19.9791 13.915C20.4515 14.2534 20.7811 14.7555 20.9037 15.3235C21.0263 15.8915 20.9333 16.4848 20.6426 16.9879C20.3519 17.4911 19.8844 17.8681 19.3311 18.0455C18.7777 18.223 18.1782 18.1883 17.6491 17.948C17.3083 17.7893 16.9343 17.715 16.5588 17.7314C16.1832 17.7478 15.8171 17.8545 15.4915 18.0424C15.1659 18.2302 14.8904 18.4938 14.6882 18.8107C14.486 19.1276 14.3632 19.4886 14.3301 19.863C14.275 20.4427 14.0057 20.981 13.575 21.3727C13.1442 21.7645 12.5828 21.9816 12.0006 21.9816C11.4183 21.9816 10.8569 21.7645 10.4262 21.3727C9.99539 20.981 9.72616 20.4427 9.67106 19.863C9.638 19.4884 9.51516 19.1273 9.31293 18.8103C9.11069 18.4933 8.83503 18.2296 8.50929 18.0418C8.18355 17.8539 7.81733 17.7472 7.44164 17.7309C7.06595 17.7146 6.69186 17.7891 6.35106 17.948C5.82195 18.1883 5.22239 18.223 4.66906 18.0455C4.11573 17.8681 3.64823 17.4911 3.35754 16.9879C3.06685 16.4848 2.97377 15.8915 3.09642 15.3235C3.21907 14.7555 3.54866 14.2534 4.02106 13.915C4.32868 13.6992 4.57979 13.4124 4.75315 13.079C4.92651 12.7456 5.01701 12.3753 5.01701 11.9995C5.01701 11.6237 4.92651 11.2535 4.75315 10.9201C4.57979 10.5866 4.32868 10.2999 4.02106 10.084C3.54932 9.74547 3.22031 9.24362 3.09796 8.67601C2.97561 8.1084 3.06867 7.51557 3.35904 7.01274C3.64942 6.50991 4.11637 6.13301 4.66915 5.95527C5.22193 5.77753 5.82104 5.81166 6.35006 6.05103C6.69082 6.20977 7.0648 6.28407 7.44036 6.26765C7.81592 6.25123 8.18199 6.14457 8.5076 5.95671C8.8332 5.76884 9.10875 5.50529 9.31093 5.18837C9.5131 4.87145 9.63594 4.51048 9.66906 4.13603" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <span className="nav-label">Settings</span>
      </Link>
    </nav>
  )
}

export default BottomNav

