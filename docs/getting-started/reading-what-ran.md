# Reading what ran

Do not infer from a timing which code stepped a run. The package says it three
ways: one line on standard error, three attributes of the driver, and a dispatch
record you can keep. Each time step is seven **sub-steps**
(<code>step_B</code>, <code>fill_B</code>, <code>update_H</code>,
<code>step_D</code>, <code>fill_D</code>, <code>update_E</code>,
<code>update_P</code>); a compiled kernel serves a sub-step it covers, and the
**array path** — the same equations in NumPy or CuPy — serves the rest. Terms
are defined in the [Glossary](../reference/glossary.md).

## The line on standard error

A run prints at most one line per process for each distinct message, beginning
`meep_gpu:`. The first run of [the quickstart](first-lift.md) on the certified
Apple host prints:

```text
meep_gpu: step path fused; 4/7 slots (step_B,update_H,step_D,update_E) via PML,fused magnetic B/H pair,offdiag; table metal; policy flush (requested flush), installed by dispatch OVERRIDING this run's own keep resolution; torch 2.10.0 certified, metal frontend metalfe-32023.850.10 certified; Apple M1 Max (applegpu_g13s) certified
```

| Part | Meaning |
|---|---|
| `step path fused` | Compiled kernels served at least one sub-step of this configuration |
| `4/7 slots (step_B,update_H,step_D,update_E)` | Four of the seven sub-steps ran as compiled kernels; the other three (here `fill_B`, `fill_D`, `update_P`) ran on the array path |
| `via PML,fused magnetic B/H pair,offdiag` | The kernels that served them. A *fused* kernel serves two consecutive sub-steps in one launch |
| `table metal` | The kernel table: `triton` or `cuda` on NVIDIA hardware, `metal` on Apple hardware. `tables cuda+triton` means both NVIDIA tables served parts of one step |
| `policy flush (requested flush), installed by dispatch` | The float32 subnormal policy the table is certified under, installed for the whole process, including the sub-steps on the array path |
| `OVERRIDING this run's own keep resolution` | Without dispatch this host would have kept subnormal numbers. This is why a default run and a `prefer_gpu=False` run can differ in the last bits ([the floating-point contract](../design/floating-point.md#the-subnormal-policy)) |
| `torch 2.10.0 certified, metal frontend ... certified` | The toolchain, read and marked: `certified`, `UNCERTIFIED`, or `uncertified-unknown` when it could not be judged. An NVIDIA run names Triton or CuPy here |
| `Apple M1 Max (applegpu_g13s) certified` | The GPU, its architecture, and the same mark: together with the toolchain, the [certified environment](../guides/kernel-dispatch.md#certification-on-an-apple-gpu). Every Apple GPU is supported, so the kernels run on it, and one is certified. On another Apple GPU it reads, for example, `Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED`, and a NOTE line follows. The certified GPU stays `certified` when PyTorch or the frontend is `UNCERTIFIED`. An Apple GPU whose architecture cannot be read (it needs macOS 14 or later) is named without one and marked `supported, uncertified-unknown`; with no name read either, it reads `device`, the reason, and `uncertified-unknown`. A GPU that is not Apple's is not marked `supported`. On NVIDIA hardware the device name appears, marked `certified` for compute capability 8.6 |

The other lines a run can print:

| Line begins | What happened | What to do |
|---|---|---|
| `meep_gpu: step path array on the host CPU; dispatch refused:` | On an Apple GPU, no compiled kernel could serve, and the whole run stepped on the host CPU. The reason follows | The run is correct. A reason ending `restricts the Metal kernels to certified environments` means `MEEP_GPU_ALLOW_UNCERTIFIED=0` is set and this Mac is outside the certified environment: unset it to run the kernels uncertified ([GPUs that are not on the certified list](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#gpus-that-are-not-on-the-certified-list)) |
| `meep_gpu: step path array; dispatch refused:` | On an NVIDIA GPU, no compiled kernel could serve; the run stepped on the array path on the device (CuPy). The reason follows | The run is correct. The reason names the device, the toolchain, the configuration, or a switch value that is not accepted (only `0` and `1` are) |
| `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host:` | The kernels ran on a device or toolchain outside the certified list: by default on a supported one (an Apple GPU outside the certified environment; an NVIDIA GPU of compute capability 7.0 to 9.0 or a Triton 3.1 release outside the certified identity); on an NVIDIA identity outside the supported range because `MEEP_GPU_ALLOW_UNCERTIFIED=1` admitted it. The line names what was read, what is certified, and why the kernels ran: on an Apple GPU, `this Apple GPU is supported`; on an NVIDIA GPU, `this NVIDIA GPU and toolchain are supported but not certified bit-identical` (`this NVIDIA GPU is`, or `this Triton version is`, when only that one was judged; under `MEEP_GPU_ALLOW_UNCERTIFIED=1`, `which also runs this NVIDIA GPU outside the supported range`) | Compare the run with a `prefer_gpu=False` run of the same simulation before relying on it. `MEEP_GPU_ALLOW_UNCERTIFIED=0` refuses such a run to the array path instead (on a Mac, the host CPU) |
| `meep_gpu: step path array (NumPy reference, prefer_gpu=False);` | `MEEP_GPU_DISPATCH` is set on a reference driver, where it does not apply | Unset it, or build the driver with `prefer_gpu=True` |
| `meep_gpu: this MEEP build is double precision and the engine steps single` | The lift read a double-precision MEEP | Expected with conda-forge MEEP; a quantity that needs more than about seven significant digits stays on MEEP |

A run that turned kernels off itself (`MEEP_GPU_DISPATCH=0` or
`MEEP_GPU_FUSED=0`) prints no step-path line, and neither does a
`prefer_gpu=False` run; the precision note above still appears when the lift
reads a double-precision MEEP.

## Three attributes of the driver

`run_on_gpu` returns a result whose `driver` is the stepped driver;
`lift_simulation` returns the driver itself. After at least one step:

| Attribute | Values |
|---|---|
| `driver.gpu` | `"cuda"`, `"metal"`, or `None` for the NumPy reference |
| `driver.active_step_path` | `"fused"` when at least one sub-step launches a compiled kernel, otherwise `"array"`. It reads `"array"` before the first step |
| `driver.fast_path_report()` | The dispatch record of this driver's configuration, a dictionary; `None` before the first step |

## The dispatch record

`fast_path_report()` is large (tens of thousands of characters on a dispatched
run). Read these keys first:

| Key | Meaning |
|---|---|
| `step_path` | `"fused"` or `"array"`, as above |
| `decision` | `"dispatched"` or `"refused"` |
| `refused_because` | The one named reason for a refusal of the whole configuration; `None` when dispatched. When it reads `no slot is left carrying a kernel`, the reasons are per sub-step, under `slots` |
| `table` | The kernel table that served: `"triton"`, `"cuda"`, `"metal"`, or `None` |
| `composition` | `dispatched_slots` and `array_slots`: which sub-steps ran as kernels and which on the array path. Present only when `decision` is `"dispatched"` |
| `slots` | Per sub-step: its `state` (`"dispatched"` or `"array"`), the kernel (`arm`) that served it, and for a sub-step on the array path its `reason`: every kernel that was considered and why it could not serve. Quote these reasons when a run was refused. Empty for a `prefer_gpu=False` driver |
| `certified` | `True` when every identity that served is certified; `False` when one that served is not, named under `uncertified["served"]` (by default on a supported identity outside the certified one, NVIDIA or Apple; on an NVIDIA identity outside the supported range under `MEEP_GPU_ALLOW_UNCERTIFIED=1`); `None` when nothing dispatched, or an identity could not be read or the certification records do not name it |
| `environment` | The device and toolchain read, and each one's verdict. On an Apple GPU: `device` (`name`, `architecture`, and the architecture again as `compute_capability`, which the certification quote reads), `device_supported` (`True` on every Apple GPU, certified or not; `False` on a GPU that is not Apple's; `None` when neither its architecture nor its name was read), `device_certified`, `torch_certified`, `frontend_certified`, `fast_math_certified`, and `recorded_environments`, what the certification records name |
| `reference_driver` | `True` for a `prefer_gpu=False` driver. Its `decision` also reads `"refused"`, so use this key to tell the reference from a GPU driver that was refused |
| `residency` | On an Apple GPU, `residency["mode"]`: `"held"` by default |

The keys `tables` and `arbitration` describe how two kernel tables share one
step on an NVIDIA host. On a host with one candidate table they read as not
reached, for example `{"not_read": "refused before any table was consulted"}`,
even when `decision` is `"dispatched"`; read `decision` and `slots` instead.

This script prints the keys above for a small run. It needs the environment of
your route ([Installation](installation.md)) and a GPU; on a host with no GPU,
pass `prefer_gpu=False` to `run_on_gpu` and read `reference_driver`:

```python
import meep as mp
from meep_gpu import run_on_gpu

sim = mp.Simulation(
    cell_size=mp.Vector3(4, 4, 4),
    resolution=8,
    boundary_layers=[mp.PML(0.8)],
    geometry=[mp.Sphere(radius=0.8, material=mp.Medium(epsilon=9))],
    sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4),
                       component=mp.Ez, center=mp.Vector3(-1.2, 0, 0))],
)
result = run_on_gpu(sim, until=5)
try:
    driver = result.driver
    report = driver.fast_path_report()
    print("gpu:", driver.gpu, " step path:", driver.active_step_path)
    for key in ("decision", "refused_because", "table", "certified", "reference_driver"):
        print(f"{key}: {report[key]}")
    composition = report.get("composition", {})    # present only when dispatched
    print("kernels:", composition.get("dispatched_slots"))
    print("array path:", composition.get("array_slots"))
    for name, slot in report["slots"].items():
        if slot.get("state") != "dispatched":
            print(f"  {name} on the array path: {slot.get('reason')}")
    print("MEEP read by the lift:", result.lift_record)
finally:
    result.close()
```

On the certified Apple host it prints `decision: dispatched`, `table: metal`,
`certified: True`, the four and three sub-steps of the line above, and a reason
for each of the three on the array path. With `prefer_gpu=False` it prints
`decision: refused`, `reference_driver: True` and `kernels: None`.

`result.lift_record` (or `driver.lift_record`) holds what the lift read from
MEEP: its version, its precision (`"single"` or `"double"`) and the precision
the engine steps in.

## Keep the record for a long run

Set `MEEP_GPU_DISPATCH_LOG` to a file path, and one JSON object, the same record,
is appended each time a configuration freezes, so the dispatch state of a long
run can be read with `tail` on the machine that runs it:

```bash
MEEP_GPU_DISPATCH_LOG=dispatch.jsonl python my_simulation.py
```

For a result you will rely on, keep the MEEP version and precision, the
preflight verdict, `driver.gpu`, and the dispatch record together with the
numbers: two runs of the same script on the same host can execute different
kernels. [Compiled-kernel dispatch](../guides/kernel-dispatch.md) covers the
switches, which table can serve your hardware, and what the coverage figures
mean.
