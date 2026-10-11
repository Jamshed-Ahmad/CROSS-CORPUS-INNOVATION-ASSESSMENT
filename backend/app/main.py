import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import router
from app.database.session import init_db
from app.logging_config import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="An Evidence-Grounded AI Framework for Cross-Corpus Innovation Assessment and Recommendation Using Research Papers and Patents"
)

# Configure CORS for React/Vite Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_PREFIX)

@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("FastAPI backend application started successfully.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
