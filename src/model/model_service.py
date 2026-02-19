import numpy as np
from pathlib import Path
from config.logger import logger
from stable_baselines3 import DQN


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "data" / "model" / "dqn_lunarlander_best_HP.zip"


def load_model():
    """
    Loads the DQN model from the specified path.
    """
    try:
        if not MODEL_PATH.exists():
            logger.error(f"❌ Critical: Model file not found at {MODEL_PATH}")
            return None

        logger.info(f"⏳ Loading model from {MODEL_PATH}...")
        model = DQN.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully!")
        return model
    
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


def predict_action(model, observation_vector: list[float]) -> int:
    """
    Performs inference using the loaded model.
    Expects a list of 8 floats and returns the action as an integer.
    """
    try:
        # Convert to numpy array and reshape for single prediction (1, 8)
        obs_array = np.array(observation_vector, dtype=np.float32).reshape(1, -1)
        
        # deterministic=True is recommended for production/evaluation
        action, _ = model.predict(obs_array, deterministic=True)
        
        # Extract scalar value from numpy array
        return int(action.item() if hasattr(action, 'item') else action)
    
    except Exception as e:
        logger.error(f"❌ Logic error during prediction: {e}")
        raise RuntimeError(f"Model inference failed: {e}")
