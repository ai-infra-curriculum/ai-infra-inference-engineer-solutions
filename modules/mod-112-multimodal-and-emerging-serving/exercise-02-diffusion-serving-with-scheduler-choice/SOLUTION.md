# Solution — Exercise 02: Diffusion Serving with Scheduler Choice

> Paired learning file: [`lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-02-diffusion-serving-with-scheduler-choice.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-02-diffusion-serving-with-scheduler-choice.md)
> Module: `mod-112-multimodal-and-emerging-serving`
> Type: implementation / decision drill

## 1. Solution overview

The learner stands up a **text-to-image diffusion** service and sweeps the
**scheduler** (sampler) choice across a fixed model and fixed prompt set. The
deliverables are: a serving configuration that lets scheduler and step count be
set per request, a latency / quality sweep across at least three scheduler
options, and a written decision for which scheduler the deployment should
default to given an explicit latency budget and acceptable-quality floor.

Three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Diffusion serving launch | Shell script / launch config | A pinned diffusion model (e.g. SDXL, SD3-medium, or FLUX-schnell) is served with scheduler and `num_inference_steps` selectable per request |
| B — Scheduler sweep | Table | At least 3 schedulers × 2 step counts, reporting per-image latency and a quality indicator; same prompts and same seed for every cell |
| C — Default-scheduler decision | 1-page memo | Names the chosen default, cites the latency budget and the quality floor it meets, and names at least one workload where another scheduler would win |

Pin the model checkpoint and the serving-framework version. Scheduler names and
default step counts shift across releases (Diffusers, StableDiffusion.cpp,
ComfyUI, vendor backends); verify every name against the pinned release.

### Why scheduler is the interesting knob

A diffusion model is trained to approximate the score of a data distribution.
Sampling turns noise into an image by solving a reverse SDE/ODE whose drift is
the learned score. The **scheduler** (also called the sampler) is the numerical
integrator for that reverse process. It picks:

1. How many discrete steps to take (`num_inference_steps`).
2. How large each step is (the timestep schedule — uniform, Karras, exponential).
3. Which integrator to use (DDIM, DDPM, Euler, Heun, DPM-Solver, DPM-Solver++,
   LMS, UniPC, LCM, TCD, ...).

Latency per image is proportional to `num_inference_steps`; each step is one
denoiser forward pass. So the practitioner question is: **for a given quality
target, which (scheduler, step count) pair gets there with the fewest steps?**

Schedulers fall into three rough buckets:

- **Baseline / older samplers** (DDIM, Euler, PNDM): robust, typically need
  25–50 steps for a well-trained base model.
- **Modern ODE solvers** (DPM-Solver++ family, UniPC): higher-order integrators
  that reach the baseline's quality in roughly half the steps (10–25) for the
  same base model.
- **Distilled-model samplers** (LCM, TCD, Lightning, Turbo, Hyper-SD, FLUX-
  schnell native): require a model that was *trained* or *distilled* to work
  with extremely low step counts (1–8). You cannot run an LCM scheduler on an
  undistilled base model and expect usable images.

The first two buckets are drop-in sampler swaps on the same model weights. The
third requires either a dedicated distilled checkpoint (FLUX-schnell, SDXL-
Lightning, SDXL-Turbo) or an LoRA adapter that was trained to make the base
model step-compatible (LCM-LoRA, Hyper-SD-LoRA).

This exercise asks the learner to characterize the first two buckets on a
single base model, and to understand when crossing into the third bucket is the
right move.

## 2. Implementation — worked walkthrough

The commands below assume a Diffusers-based serving layer (a thin HTTP wrapper
around `diffusers.AutoPipelineForText2Image`) because it exposes scheduler
selection cleanly via code. Vendor backends (TensorRT, vendor diffusion
servers, ComfyUI-as-service) follow the same pattern through their own config
files.

### 2.1 Pin the model and the schedulers under test

For Stable Diffusion XL base (undistilled):

- `stabilityai/stable-diffusion-xl-base-1.0`
- Schedulers under test: `DPMSolverMultistepScheduler`,
  `EulerAncestralDiscreteScheduler`, `UniPCMultistepScheduler`,
  `DDIMScheduler`.
- Step counts: `{20, 30}` for the ODE-solver family, `{30, 50}` for DDIM/Euler.

For a distilled comparison (optional second row of the table):

- `stabilityai/sdxl-turbo` or `ByteDance/SDXL-Lightning` LoRA on the SDXL base.
- Scheduler forced to match the distilled checkpoint's documented one.
- Step counts: `{1, 4}` for Turbo-class, `{2, 4, 8}` for Lightning.

### 2.2 Minimal serving layer

```python
# server.py — minimal diffusion HTTP server
import io, time, torch
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SCHEDULERS = {
    "ddim": DDIMScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
}

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL, torch_dtype=torch.float16
).to("cuda")

app = FastAPI()

class Req(BaseModel):
    prompt: str
    scheduler: str = "dpm++"
    num_inference_steps: int = 25
    guidance_scale: float = 7.0
    seed: int = 0

@app.post("/generate")
def generate(req: Req):
    cls = SCHEDULERS.get(req.scheduler)
    if cls is None:
        return {"error": f"unknown scheduler {req.scheduler!r}"}
    pipe.scheduler = cls.from_config(pipe.scheduler.config)
    g = torch.Generator(device="cuda").manual_seed(req.seed)
    t0 = time.perf_counter()
    out = pipe(
        prompt=req.prompt,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=g,
    )
    dt_ms = (time.perf_counter() - t0) * 1000
    buf = io.BytesIO()
    out.images[0].save(buf, format="PNG")
    return {
        "latency_ms": dt_ms,
        "scheduler": req.scheduler,
        "steps": req.num_inference_steps,
        "image_bytes_len": len(buf.getvalue()),
    }
```

Launch with `uvicorn server:app --host 0.0.0.0 --port 8020 --workers 1`. One
worker is deliberate: the pipeline holds a model's worth of GPU memory, so
multiple workers would require multiple accelerators. Concurrency is the GPU's
to handle via scheduling within one pipeline instance, not Uvicorn's.

### 2.3 Sweep harness

```python
# sweep.py
import httpx, statistics, time

PROMPTS = [
    "A watercolor painting of a red barn in a wheat field at dusk.",
    "A macro photograph of a dewdrop on a maple leaf, sharp focus.",
    "An isometric cutaway illustration of a mechanical wristwatch.",
    "A pencil sketch of a cat sleeping on a windowsill.",
    "A cyberpunk cityscape at night, neon reflections in puddles.",
]

CELLS = [
    ("ddim",    30), ("ddim",    50),
    ("euler-a", 30), ("euler-a", 50),
    ("dpm++",   20), ("dpm++",   30),
    ("unipc",   20), ("unipc",   30),
]

def one(scheduler, steps, prompt, seed):
    r = httpx.post("http://localhost:8020/generate", json={
        "prompt": prompt, "scheduler": scheduler,
        "num_inference_steps": steps, "seed": seed,
    }, timeout=180)
    r.raise_for_status()
    return r.json()["latency_ms"]

# Warm-up
one("dpm++", 10, PROMPTS[0], 0)

for sch, steps in CELLS:
    lats = []
    for i, p in enumerate(PROMPTS):
        lats.append(one(sch, steps, p, seed=1234 + i))
    print(f"{sch:<7} steps={steps:<3} "
          f"p50={statistics.median(lats):.0f}ms  "
          f"max={max(lats):.0f}ms")
```

Record outputs (as PNGs) alongside the latency so quality can be judged after
the run.

### 2.4 Quality indicator

Diffusion output quality is subjective. The exercise does not require a FID
study; it requires an **honest, labelled** quality indicator. Pick one or two
from the following and declare the choice:

- **Panel inspection.** 5 prompts × N (scheduler, steps) cells rendered to one
  contact sheet; two reviewers independently mark each cell as
  Pass / Borderline / Fail against an explicit rubric (e.g. "subject present,
  anatomy correct, no obvious artifacts, prompt-faithful").
- **CLIP score.** Automated prompt-adherence proxy. Report the mean CLIP score
  per cell. Cite it as a proxy, not as ground-truth quality — CLIP is known to
  miss fine-grained failure modes.
- **Reference comparison.** Use the model's documented default sampler at its
  documented step count as the reference cell, and judge every other cell
  relative to it (noticeably better / tie / noticeably worse).

Whichever is picked, the method note must say so explicitly. Mixing informal
"looks good to me" with CLIP scores in the same table is not a defensible
result.

### 2.5 Worked measurement table (illustrative)

Numbers below are **placeholders demonstrating expected relative ordering**;
the learner reports their own numbers.

| Scheduler | Steps | p50 latency | Panel verdict vs. reference |
|---|---|---|---|
| DDIM | 50 | *measured* | reference |
| DDIM | 30 | *measured* | noticeably worse (undercooked) |
| Euler-ancestral | 50 | *measured* | tie |
| Euler-ancestral | 30 | *measured* | borderline |
| DPM-Solver++ 2M | 30 | *measured* | tie |
| DPM-Solver++ 2M | 20 | *measured* | tie / borderline |
| UniPC | 30 | *measured* | tie |
| UniPC | 20 | *measured* | borderline |

Expected qualitative pattern on an undistilled SDXL base:

- DPM-Solver++ and UniPC reach the DDIM-50 reference's quality at roughly 20–30
  steps — the practitioner win for the ODE-solver family.
- Below ~15 steps on an undistilled base, quality degrades sharply regardless
  of scheduler. The right move for sub-10 step budgets is to switch models
  (distilled / Turbo / Lightning / schnell), not to push the undistilled base
  harder.
- Ancestral samplers (Euler-A) inject noise at each step and drift farther from
  the ODE family's trajectory, which is sometimes desirable for variety but
  hurts reproducibility.

### 2.6 Default-scheduler decision memo

A one-page memo is part of the deliverable. Example structure:

> **Chosen default.** `DPM-Solver++ 2M`, `num_inference_steps=25`.
>
> **Budget.** p50 per-image latency ≤ `<budget>` on a single
> `<accelerator>`; quality floor ≥ DDIM-50 "tie" on the internal panel for the
> prompt mix in `prompts.txt`.
>
> **Why not DDIM/Euler at 50 steps.** Meets the quality floor but exceeds the
> latency budget by `<x>%` at the same concurrency; the ODE-solver family
> matches quality in roughly half the steps.
>
> **Why not UniPC-20.** Matches DPM++ on p50 latency but `<x>/N` panel cells
> dropped to "borderline" on our fine-detail prompts; DPM++ was consistently
> "tie".
>
> **When to deviate.** For interactive thumbnail / draft modes with a <1 s
> budget, serve an SDXL-Lightning or SDXL-Turbo variant at 2–4 steps instead
> of pushing the base model's scheduler. Document as a second served model, not
> a scheduler override on the base.

The memo is short and specific. "DPM++ is faster" is not a memo; naming the
budget, the quality indicator, and the fallback is.

## 3. Validation steps

1. **Launch script runs.** `server.py` loads the pinned model, exposes
   `/generate`, and accepts all four scheduler names without error.
2. **Same seed + same scheduler + same steps is reproducible.** Two
   back-to-back calls with identical parameters return (byte-for-byte or
   image-level-identical) results modulo kernel non-determinism.
3. **Sweep covers at least three schedulers × at least two step counts.** No
   missing cells; same prompts and same seed used in every cell.
4. **Quality indicator is declared and applied.** Panel rubric, CLIP, or
   reference-comparison — explicitly named, not mixed.
5. **Latency / quality table is complete.** Every cell has both a latency and a
   quality verdict.
6. **Memo names the budget.** Not "DPM++ is best" but "DPM++ at 25 steps meets
   `<budget>` and the quality floor".
7. **Distilled fallback acknowledged.** Memo cites at least one workload where
   a distilled model, not a sampler tweak, is the right answer.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Diffusion server runs with pinned model and pinned framework version | 2 |
| 2 | Scheduler and step count are per-request, not baked into the server | 2 |
| 3 | Sweep covers ≥3 schedulers × ≥2 step counts with same prompts + same seeds | 2 |
| 4 | Quality indicator (panel / CLIP / reference) is declared and consistently applied | 2 |
| 5 | Latency + quality table is complete and reproducible | 2 |
| 6 | Memo names latency budget, quality floor, and the chosen default's justification | 2 |
| 7 | Memo cites a workload where a distilled / few-step model is the right move instead | 2 |

## 5. Common mistakes

- **Using LCM / Turbo / Lightning schedulers on an undistilled base model.**
  These samplers assume model weights trained for low-step inference. Running
  them on a vanilla SDXL base gives degraded images and misleads the sweep.
  Switch models when the step budget drops below ~10.
- **Changing scheduler and prompt and seed at the same time.** The sweep is
  only comparable when prompts and seeds are fixed. Varying all three at once
  produces aesthetic differences that cannot be attributed to the sampler.
- **Reporting single-image latency.** Latency has variance; report p50 over at
  least five prompts per cell and include max or p95 for tail visibility.
- **Confusing `num_inference_steps` with timestep schedule.** Diffusers exposes
  both; the Karras timestep spacing can be toggled independently and is itself
  a scheduler-family knob. If the exercise permits, call it out; otherwise fix
  the schedule and vary only step count.
- **Treating CLIP score as ground truth.** CLIP is a proxy with known biases
  (e.g. it rewards text-faithfulness over anatomical correctness). Report it
  but do not let it override the panel.
- **Running with `guidance_scale` drift.** Classifier-free guidance scale
  materially changes outputs; pin it per sweep. Changing it alongside the
  scheduler conflates two independent effects.
- **Reporting a scheduler win on throughput alone.** Lower steps always
  increases throughput — that is accounting, not quality. The win must be
  "fewer steps *at* quality floor", not "fewer steps full stop".
- **Omitting the warm-up.** The first generation after process start pays
  model-load, CUDA-graph-capture, and PyTorch-compile costs. Discard it.
- **Assuming one scheduler fits all prompts.** Fine-detail prompts (text in
  image, hands, mechanical cutaways) are more sampler-sensitive than
  painterly prompts. The sweep must mix prompt styles, not only one aesthetic.

## 6. References

- Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models.*
  arXiv:2006.11239 — foundational DDPM/DDIM formulation.
- Lu, C. et al. *DPM-Solver / DPM-Solver++.* arXiv:2206.00927,
  arXiv:2211.01095 — higher-order ODE solvers for diffusion sampling, the
  basis of the modern 20–30-step regime.
- Song, J., Meng, C., Ermon, S. *Denoising Diffusion Implicit Models.*
  arXiv:2010.02502 — DDIM, the baseline deterministic sampler.
- Luo, S. et al. *Latent Consistency Models.* arXiv:2310.04378 — the LCM
  distillation line, prerequisite for 2–8 step sampling.
- Hugging Face Diffusers. *Schedulers.*
  https://huggingface.co/docs/diffusers/api/schedulers/overview — practitioner
  implementation reference; names here must be pinned to the Diffusers release
  in use.
- Stability AI. *SDXL model card.*
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 — base model
  used in the walkthrough.
- NVIDIA. *TensorRT for diffusion models.*
  https://developer.nvidia.com/tensorrt — alternative practitioner
  implementation of a production diffusion server.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — diffusion outputs expose abuse surfaces (CSAM, likeness, deepfakes) not
  covered by this latency exercise; policy sits adjacent, in
  [[server-vs-edge-decision-drill]] under data-residency and in a safety
  module.
- [mitre-atlas](https://atlas.mitre.org/) — not directly invoked for the
  sampler sweep.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline for generative-image deployments.
