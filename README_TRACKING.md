# Basketball Tracking System - Player Detection & Tracking

Dit systeem is uitgebreid met player detection en tracking functionaliteit, gebaseerd op het Roboflow Basketball AI notebook.

## Nieuwe Features

### 🏀 Player Detection & Tracking
- **Player Detection**: Detecteert basketballers in real-time met YOLO
- **Multi-Player Tracking**: Volgt meerdere spelers tegelijk met unieke track IDs
- **Visual Tracking**: Elke speler krijgt een unieke kleur voor visuele identificatie

### 📡 API Server
- **Flask Backend**: RESTful API voor de React webapp
- **Video Streaming**: Live camera feed naar de webapp
- **Real-time Stats**: Automatische statistieken updates

## Installatie

### 1. Installeer Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start de API Server

```bash
python api_server.py
```

De server draait op: `http://localhost:5000`

### 3. Start de Webapp

In een nieuwe terminal:

```bash
cd webapp
npm install
npm run dev
```

De webapp draait op: `http://localhost:5173` (of een andere poort)

## Gebruik

### Standalone Tracking System

Voor direct gebruik zonder webapp:

```bash
python basketball_tracking_system.py
```

**Controls:**
- `q` - Quit/Close
- `s` - Save current frame
- `m` - Mark manual miss
- `p` - Toggle player tracking on/off

### Via Webapp

1. Start de API server (`python api_server.py`)
2. Start de webapp (`npm run dev` in webapp folder)
3. Open de webapp in je browser
4. Klik op "Start Camera" om te beginnen

## API Endpoints

### Stats & Status
- `GET /api/stats` - Haal huidige statistieken op
- `GET /api/camera/status` - Check camera status
- `GET /api/health` - Health check

### Camera Control
- `POST /api/camera/start` - Start camera en tracking
- `POST /api/camera/stop` - Stop camera
- `POST /api/reset` - Reset statistieken

### Video Feed
- `GET /api/video_feed` - Live video stream (MJPEG)

### Settings
- `GET /api/settings` - Haal huidige instellingen op
- `POST /api/settings` - Update instellingen

## Model Classes

Het YOLO model detecteert de volgende classes:
- **Class 0**: Basketball
- **Class 1**: Referee
- **Class 2**: Player (nieuw toegevoegd voor tracking)
- **Class 3**: Hoop
- **Class 4**: Ball

## Tracking Methode

Het systeem gebruikt:
1. **YOLO's Built-in Tracking**: Ultralytics YOLO heeft ingebouwde tracking functionaliteit
2. **IoU-based Matching**: Fallback tracking algoritme voor betere compatibiliteit
3. **Track Persistence**: Spelers blijven getrackt over meerdere frames

## Performance

- **FPS**: ~30 FPS op CPU, hoger op GPU
- **Resolution**: 640x480 (aanpasbaar)
- **Detection Confidence**: 0.5 (voor betere player detection)

## Troubleshooting

### Camera niet gevonden
- Controleer of de camera niet door een andere applicatie wordt gebruikt
- Probeer een andere camera index in de code

### API Server start niet
- Controleer of poort 5000 beschikbaar is
- Installeer alle dependencies: `pip install -r requirements.txt`

### Player tracking werkt niet
- Zorg dat class 2 (player) in je model zit
- Verlaag de confidence threshold als nodig
- Check of `enable_player_tracking=True` is ingesteld

## Gebaseerd op

- [Roboflow Basketball AI Detection Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb)
- Ultralytics YOLO voor object detection
- Flask voor API server

## Volgende Stappen

Mogelijke uitbreidingen:
- [ ] ByteTrack of DeepSORT voor geavanceerde tracking
- [ ] Player identificatie (jersey nummers)
- [ ] Team detection (twee teams onderscheiden)
- [ ] Bewegingsanalyse (snelheid, afstand)
- [ ] Shot analysis (shot arc, release angle)

