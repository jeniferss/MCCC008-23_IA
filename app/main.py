from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.v1 import prediction

app = FastAPI(
    title="personIA",
    description="API para análise de traços de personalidade de personagens a partir de trechos de livros",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

try:
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
except RuntimeError:
    pass

app.include_router(
    prediction.router,
    prefix="/api/v1/prediction",
    tags=["Análise de Personagens"]
)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
