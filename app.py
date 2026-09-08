"""
Phase 3: a small FastAPI backend wrapping the verified Phase 2 engine.
"""

import asyncio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from hebbian_sparse import SparseHebbianMemory, run_sparse_experiment

app = FastAPI(title="Hebbian Sandbox API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrialRequest(BaseModel):
    n: int = Field(200, ge=20, le=2000)
    sparsity: float = Field(0.05, ge=0.01, le=0.5)
    delay: int = Field(0, ge=0, le=200)
    decay_rate: float = Field(0.03, ge=0.0, le=1.0)
    noise_std: float = Field(0.22, ge=0.0, le=5.0)
    n_distractors: int = Field(0, ge=0, le=200)
    reveal_fraction: float = Field(0.4, ge=0.05, le=1.0)
    seed: int | None = None


class ExperimentRequest(BaseModel):
    n: int = Field(200, ge=20, le=2000)
    sparsity: float = Field(0.05, ge=0.01, le=0.5)
    decay_rate: float = Field(0.03, ge=0.0, le=1.0)
    noise_std: float = Field(0.22, ge=0.0, le=5.0)
    n_distractors: int = Field(0, ge=0, le=200)
    reveal_fraction: float = Field(0.4, ge=0.05, le=1.0)
    trials: int = Field(50, ge=5, le=500)
    delay_steps: list[int] = Field(default=[0, 2, 4, 6, 8, 12, 16, 20])


@app.get("/")
def serve_index():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/trial")
def trial(req: TrialRequest):
    mem = SparseHebbianMemory(req.n, sparsity=req.sparsity, seed=req.seed)
    pattern = mem.random_pattern()
    active_indices = np.where(pattern == 1)[0]
    mem.encode(pattern)

    for _ in range(req.delay):
        mem.decay(req.decay_rate, noise_std=req.noise_std)
    for _ in range(req.n_distractors):
        mem.encode(mem.random_pattern())

    cue, known_mask = mem.make_cue(active_indices, req.reveal_fraction)
    raw_activation = mem.W @ cue
    recalled = mem.recall(cue, known_mask)
    accuracy = mem.overlap_accuracy(recalled, pattern, mem.k)

    return {
        "pattern": pattern.tolist(),
        "cue": cue.tolist(),
        "known_mask": known_mask.tolist(),
        "raw_activation": raw_activation.tolist(),
        "recalled": recalled.tolist(),
        "weight_matrix": mem.W.tolist(),
        "accuracy": accuracy,
        "k_active_units": mem.k,
    }


@app.post("/experiment")
async def experiment(req: ExperimentRequest):
    results = await asyncio.to_thread(
        run_sparse_experiment,
        n=req.n,
        sparsity=req.sparsity,
        delay_steps_list=tuple(req.delay_steps),
        decay_rate=req.decay_rate,
        noise_std=req.noise_std,
        n_distractors=req.n_distractors,
        trials=req.trials,
    )
    return {"results": results}

def run_history(req: TrialRequest):
    mem = SparseHebbianMemory(req.n, sparsity=req.sparsity, seed=req.seed)
    pattern = mem.random_pattern()
    active_indices = np.where(pattern == 1)[0]
    
    raw_hist = []
    
    mem.encode(pattern)
    raw_hist.append({
        "step": 0,
        "stage": "encode",
        "weights": np.round(mem.W, 3).tolist()
    })

    for i in range(req.delay):
        mem.decay(req.decay_rate, noise_std=req.noise_std)
        raw_hist.append({
            "step": i + 1,
            "stage": "decay",
            "weights": np.round(mem.W, 3).tolist()
        })
        
    for i in range(req.n_distractors):
        mem.encode(mem.random_pattern())
        raw_hist.append({
            "step": req.delay + i + 1,
            "stage": "distractor",
            "weights": np.round(mem.W, 3).tolist()
        })

    cue, known_mask = mem.make_cue(active_indices, req.reveal_fraction)
    raw_activation = mem.W @ cue
    recalled = mem.recall(cue, known_mask)
    accuracy = mem.overlap_accuracy(recalled, pattern, mem.k)

    if len(raw_hist) > 15:
        indices = np.linspace(0, len(raw_hist) - 1, 15, dtype=int)
        hist = [raw_hist[idx] for idx in indices]
    else:
        hist = raw_hist

    return {
        "seed": req.seed,
        "pattern": pattern.tolist(),
        "cue": cue.tolist(),
        "known_mask": known_mask.tolist(),
        "raw_activation": np.round(raw_activation, 3).tolist(),
        "history": hist,
        "recalled": recalled.tolist(),
        "accuracy": accuracy,
        "k_active_units": mem.k,
    }


@app.post("/history")
async def history(req: TrialRequest):
    return await asyncio.to_thread(run_history, req)