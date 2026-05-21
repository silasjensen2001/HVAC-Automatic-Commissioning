from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Any, Literal
import traceback

from simulate import run_simulation, get_system_info
from export_plot import render_temperatures, render_valves, render_humidity, render_junctions

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


class ExportPlotRequest(BaseModel):
    tab:    Literal["temperatures", "valves", "humidity", "junctions"]
    data:   dict[str, Any]
    scheme: str = "viridis"
    title_suffix: str = ""


@app.post("/export_plot")
async def export_plot(req: ExportPlotRequest):
    try:
        renderers = {
            "temperatures": render_temperatures,
            "valves":       render_valves,
            "humidity":     render_humidity,
            "junctions":    render_junctions,
        }
        png = renderers[req.tab](req.data, req.scheme, req.title_suffix)
        return Response(content=png, media_type="image/png",
                        headers={"Content-Disposition": f'attachment; filename="hvac_{req.tab}.png"'})
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


class SystemInfoRequest(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    sim_params: dict[str, Any] = {}


@app.post("/system_info")
async def system_info_endpoint(req: SystemInfoRequest):
    try:
        result = get_system_info(req.nodes, req.edges, req.sim_params)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


@app.get("/health")
async def health():
    return {"status": "ok"}
