# F1 AI Hotlap

An interactive, reinforcement-learning powered 2D Formula 1 racing simulator where an AI agent learns to drive a Red Bull RB20 around a custom track.

## How it works

The project is split into two halves that communicate instantaneously over a WebSocket:
- **Frontend Simulator (`index.html`)**: A lightweight HTML5 Canvas 2D racing environment with physics, grass penalties, track limits, and 5 forward-facing distance sensors.
- **Python Backend (`ai/`)**: An RL agent using `stable-baselines3` (PPO) that receives the sensor data and speed, and decides how to steer and adjust the throttle. 

The project uses a **lock-step ping-pong architecture** during training—the game engine unlocks its physics from the visual 60FPS lock and calculates frames as fast as your CPU can process them, allowing the AI to experience hours of driving in under a minute!

---

## Getting Started

### Prerequisites
Make sure you have Node (or just a local web server) and Python installed.

1. Navigate to the `ai` directory:
   ```bash
   cd ai
   ```
2. Activate your virtual environment and install dependencies (if you haven't already):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install gymnasium websockets stable-baselines3
   ```

### Training the AI
To train the AI model completely from scratch:
1. Start the training server:
   ```bash
   python3 main.py
   ```
2. Open `index.html` in your web browser.
3. The AI will instantly connect and you will see the console log its high-speed training iterations (upward of 2000+ FPS). 
4. The training will automatically finish after 100,000 timesteps and save the successful model block as `f1_redbull_ai.zip`.

### Watching the AI Race
To watch the AI smoothly navigate the track using the model it just learned:
1. Stop the training script if it's still open.
2. Start the inference server:
   ```bash
   python3 play.py
   ```
3. Refresh or open `index.html` in your web browser.
4. The simulation runs at normal speeds with a visual progress tracker, showing the AI expertly hotlapping the track!

