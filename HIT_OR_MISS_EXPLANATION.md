# Hit or Miss Detection System - Uitleg

## Hoe werkt het systeem?

Het verbeterde hit-or-miss detectie systeem gebruikt een **state machine** met **trajectory tracking** om nauwkeurig te bepalen of een shot een score of miss is.

## State Machine Flow

Het systeem doorloopt de volgende states tijdens een shot:

```
IDLE → APPROACHING → NEAR_HOOP → THROUGH_HOOP → SCORED/MISSED → IDLE
```

### 1. **IDLE** (Grijs)
- Geen actieve shot gedetecteerd
- Wacht op bal beweging

### 2. **APPROACHING** (Oranje)
- Bal beweegt snel richting de ring (velocity > 5 px/frame)
- Bal is binnen 50% marge van de ring
- Systeem begint trajectory tracking

### 3. **NEAR_HOOP** (Geel)
- Bal is binnen 30% marge van de ring
- Systeem analyseert of bal door de ring gaat
- Trajectory wordt opgeslagen

### 4. **THROUGH_HOOP** (Groen)
- Bal is door de ringzone gegaan
- Systeem verifieert met trajectory analyse
- Controleert downward motion (bal valt naar beneden)

### 5. **SCORED** (Groen) of **MISSED** (Rood)
- Finale state na verificatie
- Score of miss wordt geregistreerd
- Reset naar IDLE na korte tijd

## Visuele Indicatoren op het Scherm

### Linksboven Informatie:
- **FPS**: Frames per seconde
- **Score**: Aantal scores (groen)
- **Misses**: Aantal misses (rood)
- **Accuracy**: Percentage succesvolle shots
- **State**: Huidige state van de shot (met kleur)
- **Trajectory**: Aantal punten in trajectory
- **Velocity**: Snelheid van de bal in pixels/frame

### Op het Beeld:
- **Trajectory Lijn**: Oranje lijn die de bal volgt (fading effect)
- **Velocity Arrow**: Paarse pijl die richting en snelheid toont
- **Hoop Zone**: Gekleurde rechthoek rond de ring
  - Grijs: IDLE
  - Oranje: APPROACHING
  - Geel: NEAR_HOOP
  - Groen: THROUGH_HOOP/SCORED
- **Bounding Boxes**: 
  - Oranje: Basketball
  - Geel: Hoop

## Hoe te Testen

### Test 1: Score Detectie
1. Schiet de bal door de ring
2. Kijk naar de state transitions:
   - IDLE → APPROACHING (oranje)
   - APPROACHING → NEAR_HOOP (geel)
   - NEAR_HOOP → THROUGH_HOOP (groen)
   - THROUGH_HOOP → SCORED (groen)
3. Score counter gaat omhoog
4. Trajectory lijn wordt getekend

### Test 2: Miss Detectie
1. Schiet de bal maar mis de ring
2. Kijk naar de state transitions:
   - IDLE → APPROACHING (oranje)
   - APPROACHING → NEAR_HOOP (geel)
   - Bal gaat onder de ring zonder door te gaan
   - NEAR_HOOP → MISSED (rood)
3. Miss counter gaat omhoog

### Test 3: Trajectory Tracking
1. Schiet een bal
2. Kijk naar de oranje trajectory lijn die de bal volgt
3. De lijn fade-out effect toont recente beweging
4. Velocity arrow toont richting en snelheid

## Technische Details

### Trajectory Tracking
- Slaat positie (x, y), tijd, en verticale snelheid op
- Bewaart laatste 2 seconden aan data
- Minimaal 5 punten nodig voor analyse

### Velocity Analyse
- Berekent snelheid tussen frames
- Minimum 5 px/frame nodig voor "actieve shot"
- Downward motion check: bal moet naar beneden bewegen door ring

### Miss Detectie Criteria
1. Bal was binnen 70% horizontale marge van ring
2. Bal is nu significant onder de ring (>30% ring breedte)
3. Bal was eerder boven de ring (trajectory check)
4. Bal ging voorbij ring zonder door te gaan

### Score Detectie Criteria
1. Bal is binnen 40% horizontale marge van ring
2. Bal is in verticale zone (midden tot net onder ring)
3. Bal beweegt naar beneden (downward velocity)
4. Trajectory analyse bevestigt score

## Parameters (Aanpasbaar)

In `live_camera_detection.py`:

```python
self.velocity_threshold = 5.0  # Minimum snelheid voor shot
self.hoop_zone_margin = 0.3  # 30% marge voor "near hoop"
self.score_cooldown = 1.5  # Seconden tussen scores
self.miss_cooldown = 2.0  # Seconden tussen misses
self.trajectory_history_time = 2.0  # Hoe lang trajectory bewaard wordt
```

## Troubleshooting

### Probleem: Geen scores gedetecteerd
- Check of bal goed gedetecteerd wordt (oranje box)
- Check of ring goed gedetecteerd wordt (gele box)
- Verlaag `velocity_threshold` als bal te langzaam beweegt
- Verhoog `hoop_zone_margin` voor grotere detectie zone

### Probleem: Te veel false positives
- Verhoog `score_cooldown` voor langere wachttijd
- Verlaag `hoop_zone_margin` voor kleinere zone
- Check trajectory lijn - moet logisch pad volgen

### Probleem: Misses niet gedetecteerd
- Check of bal trajectory goed getrackt wordt
- Verlaag `miss_cooldown` voor snellere detectie
- Check of bal onder ring komt (visueel controleren)

## Tips voor Beste Resultaten

1. **Goede Verlichting**: Zorg voor goede verlichting zodat bal en ring goed zichtbaar zijn
2. **Stabiele Camera**: Camera moet stabiel zijn voor betere tracking
3. **Duidelijke Shots**: Schiet duidelijk richting ring voor beste detectie
4. **Monitor State**: Kijk naar de state indicator om te zien wat het systeem detecteert
5. **Check Trajectory**: De trajectory lijn helpt te zien of tracking goed werkt

