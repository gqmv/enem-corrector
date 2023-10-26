import fastapi
from routers import essay_report_router

app = fastapi.FastAPI()

app.include_router(essay_report_router)
