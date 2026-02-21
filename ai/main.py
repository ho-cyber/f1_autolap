import asyncio
import websockets
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import threading

class F1ProEnv(gym.Env):
    def __init__(self, websocket, loop):
        super(F1ProEnv, self).__init__()
        self.websocket = websocket
        self.loop = loop
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def step(self, action):
        # We use a Future to wait for the async result in a sync way
        future = asyncio.run_coroutine_threadsafe(self._step_async(action), self.loop)
        return future.result()

    async def _step_async(self, action):
        # 1. Send action
        await self.websocket.send(json.dumps({
            "steer": float(action[0]), 
            "throttle": float(action[1])
        }))
        
        # 2. Receive state
        data = await self.websocket.recv()
        state = json.loads(data)
        
        # 3. Process
        obs = np.array(state['sensors'], dtype=np.float32) / 350.0
        speed = state.get('speed', 0)
        observation = np.concatenate([obs, [speed / 20.0]])
        
        on_track = state.get('onTrack', True)
        
        if on_track:
            if speed < 1.0:
                # Penalize stopping or reversing heavily to avoid donuts/getting stuck
                reward = -2.0 
            else:
                reward = float(speed * 1.5)
                # Slightly penalize steering to encourage straight lines if possible
                reward -= abs(float(action[0])) * 0.5
        else:
            reward = -50.0 # Huge penalty for crashing off track
        
        return observation, reward, not on_track, False, {}

    def reset(self, seed=None, options=None):
        future = asyncio.run_coroutine_threadsafe(self._reset_async(), self.loop)
        return future.result()

    async def _reset_async(self):
        # Drain old messages asynchronously if any
        try:
            while True:
                await asyncio.wait_for(self.websocket.recv(), timeout=0.001)
        except:
            pass

        await self.websocket.send(json.dumps({"command": "reset"}))
        # Receive state
        data = await self.websocket.recv()
        state = json.loads(data)
        obs = np.array(state['sensors'], dtype=np.float32) / 350.0
        speed = state.get('speed', 0)
        observation = np.concatenate([obs, [speed / 20.0]])
        return observation, {}

def train_model(env, model):
    """Function to run in a separate thread"""
    print("🚀 Training Started - AI is taking control...")
    model.learn(total_timesteps=100000, log_interval=1)
    model.save("f1_redbull_ai")
    print("💾 Model Saved!")

async def run_ai(websocket):
    print("✅ Red Bull RB20 Connected!")
    loop = asyncio.get_running_loop()
    
    # Initialize Env and Model
    env = F1ProEnv(websocket, loop)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512)
    
    # Start training in a background thread so the WebSocket loop stays free
    thread = threading.Thread(target=train_model, args=(env, model), daemon=True)
    thread.start()
    
    # Keep the connection alive while the thread runs
    while thread.is_alive():
        await asyncio.sleep(1)

async def main():
    async with websockets.serve(run_ai, "localhost", 8765):
        print("📡 Server waiting on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping AI...")