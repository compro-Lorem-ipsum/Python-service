from fastapi import FastAPI

from api.routes import router
from core.container import container


def create_app():
    app = FastAPI(title="Face Recognition API")
    app.include_router(router)

    @app.on_event("startup")
    async def startup_event():
        container.startup()

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready():
        milvus_health = container.health()
        return {
            "status": "ready" if milvus_health.get("success") else "not-ready",
            "milvus": milvus_health,
        }

    return app
