# random-walk-agent

**Random walks on agent networks with cover time, mixing time, and hitting probability analysis.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Random walks on graphs are a fundamental stochastic process with applications spanning network analysis, Markov chain theory, algorithms, and statistical physics. A random walk agent moves through a graph, at each step transitioning to a neighboring node according to a probability distribution.

This crate provides tools for:

- **Simulating** random walks on graphs with arbitrary transition probabilities
- **Analyzing** walk history: visit counts, first return times, coverage
- **Estimating** cover time (expected steps to visit all nodes)
- **Computing** mixing time (time to reach stationary distribution)
- **Computing** hitting probabilities (probability of reaching node j from i)

## Features

- **`RandomWalk`** — Agent performing random walks with configurable transition matrix
- **`WalkHistory`** — Track visited nodes, first visit times, return times, and coverage
- **`CoverTime`** — Estimate expected cover time via Monte Carlo simulation
- **`MixingTime`** — Compute mixing time to approximate stationary distribution
- **`HittingProbability`** — Compute probability of reaching targets via value iteration

## Installation

```toml
[dependencies]
random-walk-agent = "0.1.0"
```

## Quick Start

```rust
use random_walk_agent::*;

// Create a random walk on a complete graph K4
let adjacency = vec![
    vec![1, 2, 3], // node 0 connects to 1, 2, 3
    vec![0, 2, 3], // node 1
    vec![0, 1, 3], // node 2
    vec![0, 1, 2], // node 3
];
let rw = RandomWalk::from_adjacency(adjacency);

// Transition probabilities are uniform: P(0→1) = 1/3
assert!((rw.probability(0, 1) - 1.0/3.0).abs() < 1e-10);

// Compute stationary distribution (should be uniform)
let pi = rw.stationary_distribution(1000);
for &p in &pi {
    assert!((p - 0.25).abs() < 1e-6);
}

// Simulate a walk
let path = rw.walk(0, 100, 42);
assert_eq!(path.len(), 101);
assert_eq!(path[0], 0);
```

## Walk History Analysis

```rust
use random_walk_agent::*;

// Track walk statistics
let path = rw.walk(0, 500, 12345);
let history = WalkHistory::new(path, 4);

// Visit counts
let counts = history.visit_counts();
println!("Visits to each node: {:?}", counts);

// First visit times
let fvt = history.first_visit_times();
for (node, time) in fvt.iter().enumerate() {
    println!("Node {} first visited at step {:?}", node, time);
}

// First return to start
if let Some(return_time) = history.first_return_time() {
    println!("First return to start: step {}", return_time);
}

// Coverage: fraction of nodes visited
println!("Coverage: {:.1}%", history.coverage() * 100.0);
```

## Cover Time

The cover time is the expected number of steps for a random walk to visit every node at least once.

```rust
use random_walk_agent::*;

let rw = RandomWalk::from_adjacency(my_graph);
let ct = CoverTime::new(&rw);

// Estimate cover time from node 0, averaging over 1000 trials
let avg_cover = ct.estimate(0, 1000, 10000);
println!("Expected cover time from node 0: {:.1} steps", avg_cover);
```

**Known results**: For a complete graph Kₙ, the expected cover time is approximately `n · Hₙ₋₁` where `H` is the harmonic number.

## Mixing Time

The mixing time measures how quickly a random walk converges to its stationary distribution.

```rust
use random_walk_agent::*;

let rw = RandomWalk::from_adjacency(my_graph);
let mt = MixingTime::new(&rw);

// Find mixing time with ε = 0.01
let mix_time = mt.compute(0.01, 1000);
println!("Mixing time (ε=0.01): {} steps", mix_time);

// Total variation distance
let d1 = vec![1.0, 0.0, 0.0];
let d2 = vec![0.33, 0.33, 0.34];
let tv = MixingTime::total_variation(&d1, &d2);
println!("Total variation distance: {:.4}", tv);
```

## Hitting Probability

```rust
use random_walk_agent::*;

let rw = RandomWalk::from_adjacency(my_graph);
let hp = HittingProbability::new(&rw);

// Probability of ever hitting node 3 starting from node 0
let p = hp.hitting_probability(0, 3, 1000);
println!("P(hit node 3 | start at 0) = {:.4}", p);

// For irreducible chains, this should be 1.0
```

## Custom Transition Matrices

```rust
use random_walk_agent::*;

// Create a biased random walk
let rw = RandomWalk::new(vec![
    vec![0.0, 0.8, 0.2], // from 0: 80% chance to 1, 20% to 2
    vec![0.3, 0.0, 0.7], // from 1: 30% to 0, 70% to 2
    vec![0.5, 0.5, 0.0], // from 2: 50/50 to 0 or 1
]);

// The stationary distribution accounts for the bias
let pi = rw.stationary_distribution(10000);
println!("Stationary distribution: {:?}", pi);
```

## API Reference

| Type | Key Methods | Description |
|------|-------------|-------------|
| `RandomWalk` | `new`, `from_adjacency`, `walk`, `stationary_distribution` | Random walk agent |
| `WalkHistory` | `visit_counts`, `first_visit_times`, `first_return_time`, `coverage` | Walk statistics |
| `CoverTime` | `estimate` | Monte Carlo cover time estimation |
| `MixingTime` | `compute`, `total_variation` | Mixing time via power iteration |
| `HittingProbability` | `hitting_probability`, `hit_before_return` | Target hitting probabilities |

## Mathematical Background

### Stationary Distribution
For an irreducible, aperiodic chain with transition matrix P, the stationary distribution π satisfies π = πP. We compute this via power iteration.

### Cover Time
The cover time C(G) = E[max over all v of T_v] where T_v is the first hitting time of v. We estimate this via Monte Carlo.

### Mixing Time
τ(ε) = min{t : max_i ||P^t(i,·) - π||_TV < ε}. We iterate from a point mass and check total variation distance at each step.

### Hitting Probability
h(i) = P_i(hit target before source). Computed via value iteration: h(i) = Σ_j P(i,j)·h(j) with h(target)=1, h(source)=0 (for hit-before-return).

## License

MIT License. See [LICENSE](LICENSE) for details.
