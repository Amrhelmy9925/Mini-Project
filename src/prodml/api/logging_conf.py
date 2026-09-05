# import __future__
import logging 
import contextvars ,datetime,json,sys

correlation_id_var =contextvars.ContextVar("correlation_id",default="-")

class CorrelationIdFilter(logging.Filter):
    def filter(self,record):
        record.correlation_id = correlation_id_var.get()
        return True

class JSONFormatter(logging.Formatter):
    def format(self,record):
        return json.dumps({
    "timestamp" : datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat().replace("+00:00", "Z"),
    "level":record.levelname,
    "logger":record.name,

    "message":record.getMessage(),
    "correlation_id":str(getattr(record,"correlation_id","-"))},ensure_ascii=False)

def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure root logger to emit JSON to stdout.

    Safe to call multiple times (clears old handlers).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    handler.addFilter(CorrelationIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Make sure uvicorn / fastapi loggers propagate through our handler
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True