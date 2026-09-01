import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from prodml.logging_config import set_correlation_id, reset_correlation_id  # wherever these live


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "X-Correlation-ID"):
        self.header_name = header_name
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get(self.header_name, str(uuid.uuid4()))
        token = set_correlation_id(correlation_id)
        try:
            response = await call_next(request)
            response.headers[self.header_name] = correlation_id
            return response
        finally:
            reset_correlation_id(token)