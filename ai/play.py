import asyncio
import websockets
import threading
from stable_baselines3 import PPO
from main import F1ProEnv

def play_model(env, model):
    print("🚀 Inference Started - AI is driving!")
    obs, _ = env.reset()
    while True:
        # deterministic=True makes the AI pick the best action without random exploration
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Add a sleep to slow down Inference so the user can watch it!
        import time
        time.sleep(1/60) # roughly 60 FPS visual
        
        if terminated or truncated:
            obs, _ = env.reset()

async def run_inference(websocket):
    print("✅ Red Bull RB20 Connected for Inference!")
    loop = asyncio.get_running_loop()
    
    # Initialize Env
    env = F1ProEnv(websocket, loop)
    
    # Load Model
    try:
        model = PPO.load("f1_redbull_ai", env=env)
        print("💾 Model successfully loaded!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Please ensure you have trained the model first using 'python main.py'")
        return
        
    # Start inference in a background thread so the WebSocket loop stays free
    thread = threading.Thread(target=play_model, args=(env, model), daemon=True)
    thread.start()
    
    # Keep the connection alive while the thread runs
    while thread.is_alive():
        await asyncio.sleep(1)

async def main():
    async with websockets.serve(run_inference, "localhost", 8765):
        print("📡 Inference Server waiting on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping Inference...")
