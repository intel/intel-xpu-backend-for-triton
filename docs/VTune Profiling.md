This recipe explains how to do kernel profiling using Intel® VTune™ Profiler. Normally, you use Intel® VTune™ Profiler for deeper understanding of kernel’s bottleneck and possible optimizing direction, while PyTorch profiler is used for the end-to-end model bottleneck.

- [Server Setup](#server-setup)
- [Profiler Configuration](#profiler-configuration)
  - [Configure Analysis](#configure-analysis)
    - [Where Setup](#where-setup)
    - [What Setup](#what-setup)
    - [How Setup](#how-setup)
    - [Controling Profiler Range](#controling-profiler-range)
    - [Start Profiling](#start-profiling)
- [Analyze Result](#analyze-result)


# Server Setup
The full install guide could be found in [Intel® VTune™ Profiler Installation Guide](https://www.intel.com/content/www/us/en/docs/vtune-profiler/installation-guide/2023-2/overview.html). If you are using a server, the whole installation and usage process will happen on the server, your local machine does not need anything. This recipe will use the [Web Server Interface](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/web-server-ui.html) for profiling.

You can start a web interface with the following command:

```Bash
# First source the necessary env
source  ~/intel/oneapi/vtune/latest/env/vars.sh
# (Recommended) Start the VTune Profiler Server if you wish to use this server for profiling
vtune-server --web-port=8080 --data-directory ./ --allow-remote-access --enable-server-profiling
# Start the VTune Profiler Server locally
vtune-backend --web-port 8080 --data-directory ./ --allow-remote-access
```

The above command enables the remote access with the port 8080. It will normally have the output like below:

```
VTune Profiler GUI is accessible via https://1.2.3.4:8080/
```

Optionally, you can add `--data-directory="my_dir"` to specify the data directory.

# Profiler Configuration

![Welcome Page](imgs/Profiling/vtune_welcome.png)

When entered in the web interface as is shown in the above figure, it is recommended to first create a new project for each kernel, and then configure analysis.

Normally, the Triton kernel will be compiled from the aten, with the following pattern:


```Python
# triton_kernel.py

# Module definition
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, args):
        # ...
        result_tensor = ...
        return (result_tensor,)

# Case1: Compile with torch.compile
compiled = torch.compile(Repro().forward)
compiled(args)

# Case2: Compile generated with TORCH_COMPILE_DEBUG enabled
mod = make_fx(Repro(), tracing_mode='real')(args)
from torch._inductor.compile_fx import compile_fx_inner
compiled = compile_fx_inner(mod, args)
compiled(args)
```

The above Case 1 and Case 2 does not have much difference, they both will compile the module into Triton kernels. This recipe specifis on how to profile with the `compiled(args)` kernel call in the above.


## Configure Analysis

By clicking the above figure's `Configure Analysis` button, the first step is to deploy the VTune Profiler Agent.

### Where Setup

![Where Setup](imgs/Profiling/vtune_where.png)

You could use the `where` tab to deploy the VTune Profiler to a remote server.
Enter the server ip address and username / password. Then by clicking the `Deploy Agent`, it will automatically deploy the agent. After the agent is setup, now the second column of `what` is available.

### What Setup
![What Setup](imgs/Profiling/vtune_what.png)

It is recommended to wrap Python script and environment settings in a `run.sh` file, and VTune will run the file using `bash run.sh`. An example of `run.sh` is like below:

```Bash
#! /bin/bash

# 1. Source necessary env variables
source ~/env_triton.sh
# 2. Set ZE_AFFINITY_MASK for multi-card GPU. The example shows to use the 2nd card.
# Note that the python should use your conda env's triton
ZE_AFFINITY_MASK=1 /home/user/miniconda3/envs/triton_env/bin/python triton_kernel.py
```
For the `what` setup, the *Application* should set as `bash` and *Application parameters* set as your script name (`run.sh`). The working directory also need to setup to the `triton_kernel.py`'s location.

In the above fiture, step 4 and step 5 are optional, please set with your need.


### How Setup

![How Setup](imgs/Profiling/vtune_how.png)

For GPU kernel profiling, it is recommended to select the analysis type as **GPU Compute/Media Hotspots**. After selection, the configuration is shown in the above figure.

By default, there are a few available selections, such as which GPU to profile on, and GPU sampling interval. If you don't need other settings, then using these selections would be enough. Note that the target GPU should match with the above `ZE_AFFINITY_MASK`'s GPU.

The grey options is only available with custom choice. Please follow the "step 3" in the above picture, to create a new custom type. Then you can modify the options like "analyze user tasks,events, and counters" in "step 5".

### Controling Profiler Range
It is recommended to pause the profiler until the kernel code is actually running, so to reduce profiler size and generate more readable result.

For Python, one tool called [itt-python](https://github.com/NERSC/itt-python) could help to mark the start/end of the interest code. The usage is as follows:


```Python
# ... uninterested code
import itt

# warmup iters.
for i in range(50):
    compiled(args)
    torch.xpu.synchronize()

# Start Profiling
itt.resume()
compiled(args)

torch.xpu.synchronize()
# Fully detach all the collectors when done
itt.detach()

# ... uninterested code
```

### Start Profiling

![run](imgs/Profiling/vtune_run.png)
On the bottom of the web interface, there are two buttons, it is highly recommended to **Start Paused**.

Directly click on Start also works, but the resulting data would be large and noisy. By using the **Start Paused**, the collector will not collect data until be triggered by the `itt.resume()` in the above call.

# Analyze Result

After the result is collected, there are various of infos could be explored. Here is one example of identifying the optimizing direction for a kernel:

![memory_hierachy_diagram](imgs/Profiling/vtune_result_1.png)

For example, when clicked on **Graphics=>Memory Hierarchy Diagram**, there are several overview for the kernel.

By first selecting the kernel, the figure will be automatically updated and several intuitions could be made.

For example, in the above figure, Point 3 indicates that this kernel has high XVE threads occpancy, this is good for a kernel. Indicating that this kernel occupies the XPU's capacity.

However, there is one red field indicating Point 2, that the stalled time is high. By checking the Point 4, the GPU Barrier is high. Then there may have some instructions like synching that would make the XPU idle. With this in mind, then could double click the kernel and check with the assembly code.

One could also find interesting point with **Graphis=>Platform** tab as is shown below. This one is much similar with chrome tracing but with richer information. One tip is to *Zoom In and Filter in by Selection* for interesting kernel, and then find possible optimizing field.

![platform_graph](imgs/Profiling/vtune_result_2.png)
