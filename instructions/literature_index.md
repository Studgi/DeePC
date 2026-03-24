# Thesis Project & Literature Index: Data-Driven Predictive Control (DeePC)

## Next Steps Guidance
**IMPORTANT**: To determine the concrete next steps, milestones, and planning for this thesis project, you must consult the roadmap document located at: `instructions/Roadmap.pdf`. 

---

## Part 1: Thesis Project Description
*Extracted from `initial_prompt.txt`*

**Student Background**: Computer Science & Engineering

### Motivation
Data-driven predictive control (DPC) methods, such as DeePC, enable the design of controllers directly from measured input-output data without requiring an explicit system model. These approaches rely on the fundamental lemma, which guarantees that sufficiently rich data from a linear time-invariant system can represent all possible system trajectories. However, many real-world systems — including vehicles — exhibit significant nonlinear behavior, which limits the applicability of standard DPC methods. At the same time, for many systems, partial structural knowledge is available in the form of first-principles models that capture known nonlinearities. This thesis investigates whether such knowledge can be systematically integrated into the DPC framework to improve performance.

### Objective
The goal of this thesis is to explore how known structural knowledge of a system can be incorporated into a data-driven predictive control scheme. Specifically, the kinematic bicycle model will be used as a source of structural knowledge for a vehicle trajectory tracking problem. The central idea is to use known nonlinearities from the bicycle model to transform the system data into a space where the dynamics are more amenable to the linear assumptions underlying DPC. The student will implement and evaluate this approach, comparing it against standard DPC applied to the untransformed system.

### Approach
The student will begin by familiarizing themselves with the DeePC framework and the kinematic bicycle model. An existing DeePC implementation will be provided as a starting point. The main task is to design and implement physics-informed coordinate or signal transformations — derived from the known structure of the bicycle model — that reduce the effective nonlinearity of the system. DPC is then applied in the transformed space. The choice and design of suitable transformations is a key part of the thesis work and offers room for the student’s own ideas. The approach can be related to concepts from Hammerstein-Wiener systems or Koopman-based methods, where the system is represented in a space that is more linear.

### Evaluation
The evaluation will be carried out in simulation using the kinematic bicycle model. The student will assess the benefit of integrating structural knowledge along two dimensions: 
1. **Open-loop prediction accuracy**: measuring how well the DPC predictor can forecast future trajectories.
2. **Closed-loop tracking error**: measuring how well the resulting controller tracks a reference trajectory. 

Both metrics will be compared between the standard DPC formulation and the proposed structure-informed variant. The evaluation should cover a range of operating conditions, including scenarios where the linearity assumption of standard DPC is expected to break down (e.g., large steering angles, high curvature paths).

### Prerequisites
- Basic knowledge of control theory and optimization.
- Familiarity with Python or MATLAB.
- Interest in data-driven methods.

---

## Part 2: Literature Index (Nonlinear Extensions & DeePC)

This index summarizes the core methodologies, assumptions, advantages, and limitations of the key papers, allowing rapid access to the most vital elements without parsing the full papers repeatedly.

### Group 5: Core DeePC and Robust Formulations

**5.a. In the Shallows of the DeePC (Coulson et al., 2019)**
- **Core Concept**: Introduces Data-Enabled Predictive Control (DeePC). Replaces the explicit system model in MPC with a non-parametric model based on the Fundamental Lemma by Willems et al.
- **Mechanism**: Predicts future trajectories by finding a linear combination of historical, persistently exciting input/output data sequences (Hankel matrices). 
- **Assumptions**: The true underlying system is Linear Time-Invariant (LTI) and the collected data is persistently exciting.
- **Key Equations**: Uses the constraint map bounding past states ($ini$) to future predictions ($f$) via the data vector $g$.
- **Limitations**: Performs poorly in the presence of measurement noise or nonlinearities since the exact linear combination $g$ overfits the noise.

**5.b. Regularized and Distributionally Robust DeePC (Coulson et al., 2019)**
- **Core Concept**: Extends 5.a to handle stochastic measurement noise and bounded disturbances.
- **Mechanism**: Introduces a regularization term on the data combination vector $g$ (e.g., 1-norm or 2-norm penalties) and reformulates the optimization as a distributionally robust optimization problem.
- **Advantage**: Prevents overfitting to the specific noise realization in the Hankel matrix, dramatically improving robust performance in realistic stochastic settings.

**5.c. Data-Driven MPC With Stability and Robustness Guarantees (Berberich et al., 2021)**
- **Core Concept**: Bridges the gap between purely empirical DeePC and rigorous control theory by providing formal stability guarantees.
- **Mechanism**: Uses a terminal cost and terminal equality constraints constructed entirely from data (rather than a known terminal invariant set/LYAPUNOV function).
- **Advantage**: Proves that if the data is sufficiently persistently exciting and slack variables/regularization are carefully tuned, the closed-loop data-driven system guarantees asymptotic stability.

### Group 6: Nonlinear Extensions and Sampling-Based DPC

**6.a. Less is More: Contextual Sampling for Nonlinear DPC (Beerwerth & Alrifaee, 2025)**
- **Core Concept**: Addresses the failure of LTI-based DeePC on highly nonlinear systems by dynamically sampling data points locally relevant to the current state/context.
- **Mechanism**: Instead of using a single global Hankel matrix that violates the fundamental lemma (since the system isn't LTI), the algorithm continuously collects data and filters/samples a local library of trajectories that are "close" to the current operating regime (the *context*).

**6.b. Choose Wisely: Data-driven Predictive Control for Nonlinear Systems Using Online Data Selection (Näf et al., 2025)**
- **Core Concept**: Emphasizes an online metric for actively selecting the subset of historical data most representative of the upcoming nonlinear dynamics. 
- **Mechanism**: Uses similarity measures (like comparing recent past trajectories to segments in the data vault) to compose a localized Hankel matrix iteratively. By ensuring the data matrix only contains trajectories from a pseudo-linear neighborhood, standard DeePC works well.
- **Advantage**: Prevents dimension explosion in the Hankel matrix while mitigating the model mismatch induced by strong nonlinearities. 

**6.c. Linear Tracking MPC for Nonlinear Systems - The Data-Driven Case (Berberich et al., 2022)**
- **Core Concept**: Analyzes the specific problem of tracking time-varying trajectories in nonlinear systems using locally linear data-driven models. 
- **Motivation**: Provides theoretical bounds on the tracking error when applying linear data-driven techniques to a truly nonlinear system. Sets the baseline that the structural knowledge transformations aim to beat.

---

## Part 3: Synthesis with Thesis (Structural Knowledge in DeePC)

Your thesis project operates exactly at the intersection of Group 5 (Rigorous DeePC) and Group 6 (Coping with Nonlinearities).

1. Where **Group 6 (6.a, 6.b)** tries to solve the nonlinear problem by **ignoring far-away data** and building *local* linear matrices...
2. Your thesis tries to solve the nonlinear problem by **modifying the data itself globally** using *structural knowledge* (e.g., transforming the bicycle model's coordinates so the global data looks more linear, akin to Koopman or Hammerstein-Wiener concepts).

**Usage Note**:
- Keep the discrepancy between your approach (transform data to fit linear model) vs the literature baseline (restrict data to local linear subsets) clear during evaluation.
- Refer back to `instructions/Roadmap.pdf` for execution steps!