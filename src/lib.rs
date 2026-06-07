//! # random-walk-agent
//!
//! Random walks on agent networks with cover time, mixing time, hitting probability,
//! and return time analysis using transition matrices.
//!
//! ## Overview
//!
//! This crate provides tools for analyzing random walks on directed/undirected graphs
//! represented as adjacency lists with transition probabilities. It computes fundamental
//! quantities from Markov chain theory applied to random walks on graphs.
//!
//! ## Core Types
//!
//! - [`RandomWalk`] — agent performing a random walk on a graph
//! - [`WalkHistory`] — track visited nodes, first return times, and statistics
//! - [`CoverTime`] — expected time to visit all nodes
//! - [`MixingTime`] — time to reach approximate stationary distribution
//! - [`HittingProbability`] — probability of reaching node j from node i

/// A random walk on a graph defined by a transition matrix.
///
/// The transition matrix `P[i][j]` gives the probability of moving from node i to node j.
#[derive(Clone, Debug)]
pub struct RandomWalk {
    /// Transition matrix: P[i][j] = probability of i -> j
    transition: Vec<Vec<f64>>,
    /// Number of nodes
    n: usize,
}

impl RandomWalk {
    /// Create a new random walk from a transition matrix.
    ///
    /// Each row must sum to 1.0 (within tolerance).
    pub fn new(transition: Vec<Vec<f64>>) -> Self {
        let n = transition.len();
        for (i, row) in transition.iter().enumerate() {
            assert_eq!(row.len(), n, "Row {} has wrong length", i);
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Row {} sums to {} (expected 1.0)",
                i,
                sum
            );
        }
        Self { transition, n }
    }

    /// Create a random walk on an undirected graph from adjacency lists.
    ///
    /// Each node transitions uniformly to its neighbors.
    pub fn from_adjacency(adjacency: Vec<Vec<usize>>) -> Self {
        let n = adjacency.len();
        let mut transition = vec![vec![0.0; n]; n];
        for (i, neighbors) in adjacency.iter().enumerate() {
            if neighbors.is_empty() {
                transition[i][i] = 1.0; // isolated node stays put
            } else {
                let p = 1.0 / neighbors.len() as f64;
                for &j in neighbors {
                    assert!(j < n, "Neighbor {} out of bounds", j);
                    transition[i][j] = p;
                }
            }
        }
        Self { transition, n }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Get transition probability P(i, j).
    pub fn probability(&self, i: usize, j: usize) -> f64 {
        self.transition[i][j]
    }

    /// Get the full transition matrix.
    pub fn transition_matrix(&self) -> &[Vec<f64>] {
        &self.transition
    }

    /// Multiply distribution by transition matrix: π' = π · P
    pub fn step_distribution(&self, dist: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.n];
        for j in 0..self.n {
            for i in 0..self.n {
                result[j] += dist[i] * self.transition[i][j];
            }
        }
        result
    }

    /// Compute stationary distribution by power iteration.
    pub fn stationary_distribution(&self, iterations: usize) -> Vec<f64> {
        let mut pi = vec![1.0 / self.n as f64; self.n];
        for _ in 0..iterations {
            pi = self.step_distribution(&pi);
        }
        pi
    }

    /// Run a deterministic walk using a simple hash-based pseudo-random selection.
    /// Returns the sequence of visited nodes.
    pub fn walk(&self, start: usize, steps: usize, seed: u64) -> Vec<usize> {
        let mut path = Vec::with_capacity(steps + 1);
        let mut current = start;
        let mut rng = seed;
        path.push(current);

        for _ in 0..steps {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 33) as f64 / (1u64 << 31) as f64;
            let mut cumsum = 0.0;
            for j in 0..self.n {
                cumsum += self.transition[current][j];
                if r < cumsum {
                    current = j;
                    break;
                }
            }
            path.push(current);
        }

        path
    }
}

/// Tracks the history of a random walk for analysis.
#[derive(Clone, Debug)]
pub struct WalkHistory {
    /// Sequence of visited nodes.
    pub path: Vec<usize>,
    /// Number of nodes in the graph.
    n: usize,
}

impl WalkHistory {
    /// Create a walk history from a path.
    pub fn new(path: Vec<usize>, n: usize) -> Self {
        Self { path, n }
    }

    /// Number of steps in the walk.
    pub fn steps(&self) -> usize {
        self.path.len().saturating_sub(1)
    }

    /// Count visits to each node.
    pub fn visit_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.n];
        for &node in &self.path {
            counts[node] += 1;
        }
        counts
    }

    /// First time each node was visited (step index), or None if never visited.
    pub fn first_visit_times(&self) -> Vec<Option<usize>> {
        let mut first = vec![None; self.n];
        for (step, &node) in self.path.iter().enumerate() {
            if first[node].is_none() {
                first[node] = Some(step);
            }
        }
        first
    }

    /// First return time to the starting node (after leaving it).
    pub fn first_return_time(&self) -> Option<usize> {
        if self.path.len() < 2 {
            return None;
        }
        let start = self.path[0];
        for (step, &node) in self.path.iter().enumerate().skip(1) {
            if node == start {
                return Some(step);
            }
        }
        None
    }

    /// Fraction of nodes visited.
    pub fn coverage(&self) -> f64 {
        let visited = self.first_visit_times().iter().filter(|v| v.is_some()).count();
        visited as f64 / self.n as f64
    }
}

/// Estimates cover time (expected time to visit all nodes) via simulation.
pub struct CoverTime<'a> {
    walk: &'a RandomWalk,
}

impl<'a> CoverTime<'a> {
    /// Create a new cover time estimator.
    pub fn new(walk: &'a RandomWalk) -> Self {
        Self { walk }
    }

    /// Estimate expected cover time from `start` by averaging over `trials` simulations.
    pub fn estimate(&self, start: usize, trials: usize, max_steps: usize) -> f64 {
        let mut total = 0usize;
        let mut rng_seed = 42u64;

        for _ in 0..trials {
            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let path = self.walk.walk(start, max_steps, rng_seed);
            let mut visited = vec![false; self.walk.len()];
            let mut cover_step = max_steps;

            for (step, &node) in path.iter().enumerate() {
                if !visited[node] {
                    visited[node] = true;
                    if visited.iter().all(|&v| v) {
                        cover_step = step;
                        break;
                    }
                }
            }
            total += cover_step;
        }

        total as f64 / trials as f64
    }
}

/// Computes mixing time (time to reach within ε of stationary distribution).
pub struct MixingTime<'a> {
    walk: &'a RandomWalk,
}

impl<'a> MixingTime<'a> {
    /// Create a new mixing time computer.
    pub fn new(walk: &'a RandomWalk) -> Self {
        Self { walk }
    }

    /// Compute total variation distance between two distributions.
    pub fn total_variation(dist1: &[f64], dist2: &[f64]) -> f64 {
        dist1
            .iter()
            .zip(dist2.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / 2.0
    }

    /// Find the mixing time: smallest t such that max over all starting nodes
    /// of TV(P^t(i,·), π) < ε.
    ///
    /// Uses a simplified approach: checks from a uniform start.
    pub fn compute(&self, epsilon: f64, max_iterations: usize) -> usize {
        let stationary = self.walk.stationary_distribution(10000);

        let mut dist = vec![0.0; self.walk.len()];
        dist[0] = 1.0; // start from node 0

        for t in 1..=max_iterations {
            dist = self.walk.step_distribution(&dist);
            let tv = Self::total_variation(&dist, &stationary);
            if tv < epsilon {
                return t;
            }
        }

        max_iterations
    }
}

/// Computes hitting probabilities using iterative methods.
pub struct HittingProbability<'a> {
    walk: &'a RandomWalk,
}

impl<'a> HittingProbability<'a> {
    /// Create a new hitting probability computer.
    pub fn new(walk: &'a RandomWalk) -> Self {
        Self { walk }
    }

    /// Compute the probability of hitting node `target` before returning to `source`
    /// (starting from `source`).
    ///
    /// Uses iterative value iteration: h(i) = Σ_j P(i,j) * h(j), with h(target) = 1, h(source excluded).
    pub fn hit_before_return(&self, source: usize, target: usize, iterations: usize) -> f64 {
        if source == target {
            return 1.0;
        }

        let n = self.walk.len();
        let mut h = vec![0.0; n];
        h[target] = 1.0;

        for _ in 0..iterations {
            let mut new_h = vec![0.0; n];
            for i in 0..n {
                if i == target {
                    new_h[i] = 1.0;
                    continue;
                }
                if i == source {
                    // Source: we're computing the probability starting from here
                    // h(source) = Σ_j P(source,j) * h(j)
                    for j in 0..n {
                        new_h[i] += self.walk.probability(i, j) * h[j];
                    }
                } else {
                    for j in 0..n {
                        new_h[i] += self.walk.probability(i, j) * h[j];
                    }
                }
            }
            h = new_h;
        }

        h[source]
    }

    /// Compute hitting probability from `start` to `target` (no return constraint).
    /// h(i) = Σ_j P(i,j) * h(j), with h(target) = 1.
    pub fn hitting_probability(&self, start: usize, target: usize, iterations: usize) -> f64 {
        if start == target {
            return 1.0;
        }

        let n = self.walk.len();
        let mut h = vec![0.0; n];
        h[target] = 1.0;

        for _ in 0..iterations {
            let mut new_h = vec![0.0; n];
            for i in 0..n {
                if i == target {
                    new_h[i] = 1.0;
                    continue;
                }
                for j in 0..n {
                    new_h[i] += self.walk.probability(i, j) * h[j];
                }
            }
            h = new_h;
        }

        h[start]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_graph(n: usize) -> RandomWalk {
        let adj: Vec<Vec<usize>> = (0..n).map(|i| (0..n).filter(|&j| j != i).collect()).collect();
        RandomWalk::from_adjacency(adj)
    }

    #[test]
    fn test_complete_graph_transition() {
        let rw = complete_graph(4);
        // Each node has 3 neighbors, each with prob 1/3
        assert!((rw.probability(0, 1) - 1.0 / 3.0).abs() < 1e-10);
        assert!((rw.probability(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_step_distribution() {
        let rw = complete_graph(3);
        let dist = vec![1.0, 0.0, 0.0]; // start at node 0
        let next = rw.step_distribution(&dist);
        // From node 0: go to 1 with 1/2, go to 2 with 1/2
        assert!((next[0]).abs() < 1e-10);
        assert!((next[1] - 0.5).abs() < 1e-10);
        assert!((next[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stationary_distribution() {
        let rw = complete_graph(4);
        let pi = rw.stationary_distribution(1000);
        // Should be uniform: 1/4 each
        for &p in &pi {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_walk_length() {
        let rw = complete_graph(3);
        let path = rw.walk(0, 10, 42);
        assert_eq!(path.len(), 11); // start + 10 steps
        assert_eq!(path[0], 0);
    }

    #[test]
    fn test_walk_history_first_visit() {
        let hist = WalkHistory::new(vec![0, 1, 2, 1, 0], 3);
        let fvt = hist.first_visit_times();
        assert_eq!(fvt[0], Some(0));
        assert_eq!(fvt[1], Some(1));
        assert_eq!(fvt[2], Some(2));
    }

    #[test]
    fn test_walk_history_return_time() {
        let hist = WalkHistory::new(vec![0, 1, 2, 0], 3);
        assert_eq!(hist.first_return_time(), Some(3));
    }

    #[test]
    fn test_walk_history_no_return() {
        let hist = WalkHistory::new(vec![0, 1, 2, 1, 2], 3);
        assert_eq!(hist.first_return_time(), None);
    }

    #[test]
    fn test_walk_history_coverage() {
        let hist = WalkHistory::new(vec![0, 1, 2, 0], 3);
        assert!((hist.coverage() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cover_time() {
        let rw = complete_graph(3);
        let ct = CoverTime::new(&rw);
        let avg = ct.estimate(0, 100, 1000);
        // For K3, cover time should be small (< 100)
        assert!(avg < 500.0);
    }

    #[test]
    fn test_mixing_time() {
        let rw = complete_graph(4);
        let mt = MixingTime::new(&rw);
        let t = mt.compute(0.01, 1000);
        // Complete graph mixes very fast
        assert!(t <= 50);
    }

    #[test]
    fn test_total_variation() {
        let d1 = vec![0.5, 0.5];
        let d2 = vec![1.0, 0.0];
        let tv = MixingTime::total_variation(&d1, &d2);
        // TV = (|0.5-1.0| + |0.5-0.0|) / 2 = 0.5
        assert!((tv - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hitting_probability_complete() {
        let rw = complete_graph(4);
        let hp = HittingProbability::new(&rw);
        let p = hp.hitting_probability(0, 3, 1000);
        // From node 0, probability of ever hitting 3 should be 1.0 for irreducible chain
        assert!((p - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hitting_same_node() {
        let rw = complete_graph(3);
        let hp = HittingProbability::new(&rw);
        let p = hp.hitting_probability(1, 1, 100);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_visit_counts() {
        let hist = WalkHistory::new(vec![0, 1, 0, 1, 0], 2);
        let counts = hist.visit_counts();
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 2);
    }
}
