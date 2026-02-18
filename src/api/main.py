from fastapi import FastAPI, HTTPException, Request
from src.model.model_service import load_model, predict_action
from src.api.schemas import Observation, PredictionResponse
from contextlib import asynccontextmanager
from config.logger import logger


@asynccontextmanager
async def lifespan(api: FastAPI):
    # Load and inject model during startup
    api.state.model = load_model()

    if api.state.model:
        logger.info("✅ Model successfully injected into app state")
    else:
        logger.error("❌ Application started WITHOUT an active model")
    yield

    # Cleanup
    logger.info("ℹ️ Application shutting down...")
    if hasattr(api.state, "model"):
        del api.state.model
    logger.info("✅ Cleanup completed")


api = FastAPI(lifespan=lifespan)


def _get_model(request: Request):
    """
    Helper function to safely retrieve the model from the application state.
    """
    return getattr(request.app.state, "model", None)


@api.get("/health")
def health():
    """
    Check the health status of the API.

    Returns:
        dict: A message confirming the API is operational.

    Raises:
        HTTPException: If an internal error occurs during the check.
    """
    try:
        return {"message": "API is running correctly"}
    except Exception as e:
        logger.error(f"❌ API Health check failed: {e}")
        raise HTTPException(status_code=500, detail="API is experiencing issues.")


@api.get("/model_status")
def get_model_status(request: Request):
    """
    Check if the model is currently loaded in the application state.
    """
    model = _get_model(request)
    is_loaded = model is not None
    logger.info(f"Model status check: {'Alive' if is_loaded else 'Not loaded'}")
    return {"is_model_loaded": is_loaded}


@api.post("/load_model")
def manual_load_model(request: Request):
    """
    Manually trigger the model loading process.
    """
    model = load_model()
    if model:
        request.app.state.model = model
        logger.info("✅ Model manually reloaded and injected")
        return {"message": "Model loaded successfully"}

    logger.error("❌ Manual model loading failed")
    raise HTTPException(status_code=500, detail="Failed to load model manually")


@api.post("/predict", response_model=PredictionResponse)
def predict(request: Request, observation: Observation):
    """
    Predict the action to take based on an environment observation vector.

    Args:
        request (Request): The request object used to access the model injected into the app state.
        observation (Observation): A Pydantic model containing the 8-value state vector.

    Returns:
        PredictionResponse: A Pydantic model containing the predicted action (0, 1, 2, or 3).

    Raises:
        HTTPException: If the model is not loaded (503) or if prediction fails.
    """
    # 1. Access the model from the application state using the helper
    model = _get_model(request)
    if not model:
        logger.error("❌ Prediction failed: Model is not loaded in api state")
        raise HTTPException(
            status_code=503,
            detail="Model is currently not loaded. Please try again later.",
        )

    try:
        # 2. delegate business logic to model_service
        action = predict_action(model, observation.state)

        # 3. Return the response using the schema
        return PredictionResponse(action=action)

    except RuntimeError as re:
        # Specific business logic error
        logger.error(f"❌ Logic error during prediction: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        # General unhandled error
        logger.error(f"❌ Unexpected prediction error: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during prediction."
        )
