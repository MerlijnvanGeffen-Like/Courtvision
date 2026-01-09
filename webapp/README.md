# Courtvision Web App

A React web application for basketball tracking, built based on the Figma design.

## Features

- **Home Screen**: Real-time score tracking, accuracy display, and camera controls
- **Stats Screen**: Current session and all-time statistics
- **Leaderboard**: Top players ranking with user position
- **Settings**: Appearance, camera settings, and device configuration

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Navigate to the webapp directory:
```bash
cd webapp
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:3000`

## Project Structure

```
webapp/
├── src/
│   ├── components/      # Reusable components (Header, BottomNav)
│   ├── pages/           # Page components (Home, Stats, Leaderboard, Settings)
│   ├── styles/          # Global styles
│   ├── App.jsx          # Main app component with routing
│   └── main.jsx         # Entry point
├── public/              # Static assets
├── index.html           # HTML template
├── package.json         # Dependencies
└── vite.config.js      # Vite configuration
```

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Preview Production Build

```bash
npm run preview
```

## Design

The app follows a dark theme design with:
- Primary gradient: Orange to Red (#ff751f to #ad0030)
- Accent color: Cyan (#1ec5ff)
- Background: Dark (#0a0a0a, #1a1a1f, #111)
- Mobile-first responsive design

## Technologies Used

- React 18
- React Router DOM
- Vite
- CSS Modules

## Notes

- The app is designed for mobile viewport (428px width)
- Camera functionality is UI-only (not connected to actual camera)
- Stats and leaderboard data are currently static/mock data

