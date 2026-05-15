from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import traceback

from simulate import run_simulation

app = FastAPI(title="HVAC Commissioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    sim_params: dict[str, Any]


@app.post("/simulate")
async def simulate(req: SimulationRequest):
    try:
        result = run_simulation(req.nodes, req.edges, req.sim_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


@app.get("/health")
async def health():
    return {"status": "ok"}
