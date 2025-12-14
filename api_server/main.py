"""
Main entry point for the FastAPI server.
"""

import uvicorn
from api_server.config import config
from api_server.app import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=config.reload,
        log_level=config.log_level.lower()
    )

