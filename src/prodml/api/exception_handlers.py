import logging
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

_log = logging.getLogger("prodml.api")

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(loc) for loc in err["loc"] if loc != "body"), "message": err["msg"]}
        for err in exc.errors()
    ]
    _log.exception("Validation error on %s %s: %s", request.method, request.url.path, errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "Validation failed", "errors": errors},
    )

async def unhandled_exception_handler(request: Request, exc: Exception):
    _log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )