import logging
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prodml.api.routes.health import router as health_router
from prodml.api.routes.predict import router as predict_router
from prodml.api.routes.metadata import router as metadata_router
from prodml.api.middleware.correlation import CorrelationIdMiddleware
from prodml.api.exception_handlers import (
    validation_exception_handler,
    unhandled_exception_handler,
)
from prodml.predict import PredictSettings, DurationPredictor
from prodml.logging_config import setup_logger


def load_predictor():
    """
    Load the trained model and vectorizer from disk.
    """

    settings = PredictSettings()
    predictor = DurationPredictor(settings)
    predictor.load()
    return predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = PredictSettings()
    setup_logger(log_level=settings.log_level, log_format=settings.log_format)
    _log = logging.getLogger("ProdML API")
    _log.info("ProdML API is starting up...")
    predictor = load_predictor()
    app.state.predictor = predictor
    yield
    _log.info("ProdML API is shutting down...")
    app.state.predictor = None

app = FastAPI(
    title="ProdML API",
    description="API for ProdML model prediction",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(metadata_router)
app.add_middleware(CorrelationIdMiddleware)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)