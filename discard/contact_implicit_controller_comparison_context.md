# Project Context: Comparative Evaluation of Contact-Implicit Controllers

## Purpose

This document gives high-level context to an AI that may modify the project's simulations, experiment scripts, parameter sweeps, visualizations, or run commands.

The central point is:

> The project's primary goal is to compare contact-implicit controllers and understand their relative strengths, weaknesses, sensitivities, and failure modes.

The simulations, sweeps, metrics, plots, and videos are not the final objective. They are the instruments used to make that comparison systematic, interpretable, and fair.

---

## Main research goal

Contact-rich controllers are often demonstrated on different robots, tasks, metrics, physical assumptions, and tuning procedures. Because of this, it is difficult to tell whether one controller is genuinely better or whether its demonstration simply favors its design.

This project creates a common evaluation framework in which multiple controllers are tested on the same underlying contact problems.

The intended result is not merely a set of successful simulations. It is a comparative explanation of:

- where each controller succeeds;
- where each controller fails;
- what contact behavior it uses;
- how robust that behavior is;
- how sensitive it is to tuning;
- how performance changes as the task becomes harder;
- whether apparent task success corresponds to physically appropriate contact behavior.

No single run can answer these questions. The project therefore relies on repeated, structured experiments.

---

## Research hierarchy

### 1. Primary goal: compare controllers

The final scientific objective is comparative understanding.

The project should reveal:

- which controllers discover useful contact modes;
- which maintain them;
- which complete the task through undesirable behavior such as repeated impacts;
- which have broad versus narrow tuning regions;
- which fail gradually versus abruptly;
- which are robust across task difficulty;
- which failures come from mode discovery, mode maintenance, instability, model mismatch, or poor tuning.

### 2. Intermediate goal: construct behavioral maps through sweeps

The project varies:

- a **task parameter**, which changes the physical difficulty of the contact problem; and
- a **controller parameter**, which changes the controller's effective behavior or tuning.

The resulting sweep maps behavior across a two-dimensional experimental space.

These maps reveal:

- broad success regions;
- narrow tuning windows;
- transition boundaries;
- unstable regions;
- qualitative changes in strategy;
- sensitivity to task difficulty;
- differences in robustness across controllers.

A single favorable tuning is not enough. The structure of the full parameter region is part of the result.

### 3. Instrumental goal: produce metrics and visualizations

The sweep must be converted into interpretable evidence.

Useful outputs include:

- task-tracking error;
- contact-mode ratios;
- heatmaps over task and controller parameters;
- state and input trajectories;
- contact timelines;
- qualitative labels such as steady pushing, tapping, striking, sticking, or sliding;
- videos or simulator views of representative trials.

These outputs support the comparison. They are not ends in themselves.

A visually impressive simulation is only useful if it helps explain controller behavior.

---

## Core experimental idea: atomic contact tasks

Complex manipulation tasks contain many contact transitions at once. When a controller fails, it may be unclear which transition caused the failure.

The project therefore decomposes contact-rich manipulation into simple "atomic" tasks. Each task isolates one contact decision or transition.

The current work emphasizes two categories.

---

## Atomic task 1: make and break contact

### Question

How far from a contact surface can the controlled object start and still approach, make useful contact, and perform the task?

### Basic setup

A one-dimensional end-effector begins at some distance from a movable block. The objective is to move the block at a desired velocity.

Appropriate success requires the end-effector to:

1. approach the block;
2. make contact;
3. avoid unnecessarily violent impact;
4. maintain useful contact;
5. push the block steadily.

The main task parameter is the initial separation between the end-effector and the block.

### Why this task matters

A controller can fail in several distinct ways:

- never approach;
- stop before contact;
- collide but fail to continue pushing;
- repeatedly strike the block;
- make and lose contact intermittently;
- remain in contact but track the target velocity poorly;
- push steadily and accurately.

A final displacement, reward, or velocity error can hide these differences.

The experiment should therefore record both task performance and contact behavior.

The real question is:

> How reliably and appropriately does each controller transition from no contact to sustained useful contact as task difficulty and tuning change?

---

## Atomic task 2: sticking and sliding

### Question

Can the controller produce the correct combination of sticking and sliding at different contact interfaces?

### Basic setup

An end-effector pushes a block while friction determines whether:

- the end-effector sticks to or slides across the block; and
- the block sticks to or slides across the ground.

The task parameter changes the frictional conditions and therefore changes how difficult it is to obtain the desired mode.

A representative desired behavior is:

- the end-effector sticks to the block;
- the block slides along the ground.

### Why this task matters

A controller may move the block while using the wrong frictional mode.

Possible outcomes include:

- the block remains stuck;
- the end-effector slides while the block moves;
- both interfaces slide;
- both interfaces stick;
- the desired stick-slide combination is maintained.

The real question is:

> Across task conditions and controller settings, which controllers can select and maintain the correct frictional mode?

---

## Role of parameter sweeps

A sweep is a structured set of repeated trials.

For each controller, the experiment varies at least:

1. a physical task parameter;
2. a controller-specific tuning parameter.

The important output is the pattern across the sweep, not an isolated number.

A good sweep can reveal:

- broad versus narrow regions of success;
- sensitivity to tuning;
- performance degradation as the task becomes harder;
- low task error paired with poor contact behavior;
- abrupt switches between strategies;
- unstable or inconsistent regimes;
- run-to-run variability.

Every trial should preserve its full parameter coordinates so that it can be reproduced, compared, and visualized.

The sweep system should support:

- rerunning one point;
- repeating points across random seeds;
- expanding parameter ranges;
- comparing equivalent points across controllers;
- selecting representative trials for closer inspection.

---

## Role of visualizations

Contact behavior often cannot be understood from a single scalar metric.

### Heatmaps

Heatmaps show how a metric or qualitative mode changes over:

- task difficulty; and
- controller tuning.

They reveal regions, boundaries, and sensitivities.

### Contact-mode plots

These indicate whether the system is:

- in or out of contact;
- sticking or sliding;
- maintaining or repeatedly losing the desired mode.

They distinguish superficially similar task outcomes.

### Trajectory plots

Position, velocity, force, and timing plots can reveal:

- delayed approach;
- impact spikes;
- oscillation;
- overshoot;
- steady-state error;
- command saturation;
- divergence;
- differences between planned and applied inputs.

### Videos and simulation viewers

Visual inspection helps validate qualitative labels and understand representative sweep points.

However, videos should support the systematic analysis, not replace it. Each selected video should correspond to a known trial with logged parameters.

---

## Fair-comparison requirements

An AI modifying simulations or commands should preserve comparability unless the requested change explicitly changes the experiment.

### Common task definition

Controllers should solve the same physical problem with the same desired outcome.

### Common initial conditions

Equivalent sweep points should use equivalent starting states.

### Common plant assumptions

Mass, geometry, friction, damping, actuator limits, contact parameters, and timing should be held constant or documented.

### Common metrics

The same task-error and contact-mode definitions should be applied across controllers.

### Comparable computational conditions

Control frequency, planning horizon, runtime limits, warm-up handling, and termination rules should be aligned where possible and reported when they cannot be identical.

### Transparent interfaces

Different controllers may require different command formats or software interfaces. Those differences should be handled explicitly rather than by silently changing the task.

### No silent stabilization

Adding damping, filtering, clipping, compliance, or force limits can change the physical problem. Such changes may be valid, but they must be visible in the experiment definition.

---

## What the simulation infrastructure is for

The simulation stack exists to support controlled comparison.

Its responsibilities include:

- representing the same atomic task for multiple controllers;
- applying controller outputs to the plant;
- exposing the current state;
- logging planned, commanded, and applied inputs;
- determining contact status;
- recording trajectories;
- supporting parameterized initial conditions;
- running repeatable trials;
- automating sweeps;
- producing outputs suitable for shared analysis.

The software architecture may differ between integrations. One controller may run in the same process as its simulator, while another may communicate with an external simulator through messages or shared memory.

Those implementation differences are secondary. The scientific question is whether the resulting trials are comparable and informative.

---

## What should be logged

A trial should retain enough information to reconstruct both task performance and contact behavior.

At minimum:

- trial identifier;
- controller identifier;
- controller parameter values;
- task parameter values;
- initial conditions;
- plant and contact parameters;
- simulation timestep;
- control timestep;
- planning horizon or duration;
- random seed, where applicable;
- state trajectory;
- planned input trajectory;
- applied input trajectory;
- contact status over time;
- task error;
- contact-mode metric;
- termination reason;
- software version or commit where possible.

For frictional tasks, state-only logs may be insufficient. The system should retain enough contact and relative-motion information to determine sticking versus sliding.

---

## Planned versus applied behavior

The project should distinguish among:

- the controller's planned future input;
- the immediate command selected from that plan;
- any interpolated, delayed, filtered, or clipped command;
- the input actually applied by the simulator.

This matters because a controller can generate a reasonable plan while the interface or timing layer applies something different.

When modifying bridges or command pipelines, do not collapse these into one signal.

---

## How to judge a modification

A simulation or command change should be evaluated by asking:

1. Does it preserve the controller-comparison objective?
2. Does it keep equivalent tasks equivalent across controllers?
3. Does it improve the reliability or interpretability of the sweep?
4. Does it preserve the ability to classify contact behavior?
5. Does it change the plant physics or only the software interface?
6. Does it alter the controller's effective tuning?
7. Are the changes reflected in logs and metadata?
8. Can old and new results still be compared?
9. Does it help identify controller differences rather than merely make one run look better?

A change that improves one controller's appearance but destroys comparability is scientifically counterproductive.

---

## Interpreting success

The strongest result is not necessarily the lowest error at one hand-selected parameter point.

A controller may compare favorably if it has:

- a broad region of acceptable performance;
- stable contact behavior;
- low sensitivity to tuning;
- consistent behavior across task difficulty;
- interpretable failure boundaries;
- physically appropriate strategies;
- reproducible results.

Conversely, low error may be misleading if the controller:

- strikes repeatedly;
- chatters between modes;
- relies on unrealistic contact behavior;
- works only in a narrow tuning region;
- varies substantially across seeds.

The evaluation should expose these distinctions.

---

## Broader direction

The current atomic tasks are foundations for a larger comparative framework.

The same logic can later extend to:

- contact-point repositioning;
- gaiting between contacts;
- multiple simultaneous contacts;
- longer planning horizons;
- larger state spaces;
- higher-precision manipulation;
- more complex contact sequences;
- additional controller integrations.

The long-term value is a reusable system in which new controllers can be evaluated under the same conceptual framework.

---

## Concise project story

Contact-implicit controllers are difficult to compare because they are usually demonstrated on different tasks with different metrics and tuning conventions.

This project evaluates multiple controllers on the same simple contact transitions. Each atomic task isolates a physical decision, such as making contact or maintaining sticking contact. The experiment varies both task difficulty and controller tuning, then records task performance and contact behavior across the resulting sweep.

The sweep maps, metrics, trajectory plots, contact classifications, and simulation visualizations are evidence-generating tools for the primary scientific objective: understanding how the controllers differ, where each works, how robustly it works, and why it fails.

When editing simulations or commands, preserve that hierarchy:

> Controller comparison is the goal. Sweeps and visualizations are the instruments.
