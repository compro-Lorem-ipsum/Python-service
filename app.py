from fastapi import FastAPI
from api.routes import router

def create_app():
    app = FastAPI(title="Face Recognition API")
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app
