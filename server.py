import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from cfr.deep_cfr import DeepCFR
from game_engine.games import LeducState

app = Flask(__name__)
CORS(app)

# Initialize solver layout for Leduc
# (Must match the dimensions trained in main.py)
solver = DeepCFR(
    game_factory     = LeducState.new_game,
    state_dim        = 30,
    num_actions      = 3,
    num_players      = 2,
    buffer_capacity  = 100,  # Unused for inference
    hidden_dim       = 64,
    device           = "cpu"
)

# Load the trained model
try:
    solver.load_model("leduc_model.pth")
    print("Successfully loaded leduc_model.pth.")
except Exception as e:
    print(f"Warning: Could not load trained model ({e}). Proceeding dynamically.")

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        s = LeducState()
        
        # 1. Provide Context for Card Distribution
        h = data.get("hole", [None, None])
        s.hole_cards = [
            (h[0]["r"], h[0]["s"]) if h[0] else None,
            (h[1]["r"], h[1]["s"]) if h[1] else None
        ]
        
        comm = data.get("community")
        s.community_card = (comm["r"], comm["s"]) if comm else None
        
        # 2. Maintain Game Value Dimensions
        s.pot = float(data.get("pot", 2.0))
        s.bets = [float(x) for x in data.get("bets", [1.0, 1.0])]
        s.street = int(data.get("street", 0))
        s.acting_player = int(data.get("acting", 1))
        
        # 3. Synchronize Actions
        s.history = data.get("history", [])

        # Get neural network strategy distribution
        policy = solver.get_policy(s, s.acting_player)
        
        probs = [
            policy.get(0, 0.0),
            policy.get(1, 0.0),
            policy.get(2, 0.0)
        ]
        return jsonify({"probs": probs})
    except Exception as e:
        print(f"Error handling predict: {e}")
        return jsonify({"probs": [0.33, 0.33, 0.34]}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
