<div align="center">

# 🔐 Privacy-Aware Embodied AI: Towards a Comprehensive Framework

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![arXiv](https://img.shields.io/badge/arXiv-2510.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2510.xxxxx)
[![GitHub stars](https://img.shields.io/github/stars/JL-F-D/Survey?style=social)](https://github.com/JL-F-D/Survey)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JL-F-D/Survey/graphs/commit-activity)

</div>

<div align="center">
  <img src="assets/banner.png" width="100%" alt="Survey Banner"/>
</div>

---

<p align="center">
  <b>🔥 This is a curated paper list of "Privacy-Aware Embodied AI: Towards a Comprehensive Framework".</b>
  <br>
  To the best of our knowledge, this work presents the <b>first comprehensive survey</b> on privacy considerations in Vision-Language-Action (VLA) systems, covering the entire closed-loop framework from instruction understanding to environment perception.
  <br>
  We will continue to <b>UPDATE</b> this repository with the latest developments! 
  <br>
  ⭐ <b>Star us to stay tuned!</b> 😘
</p>

---

## 📢 News

| Date | News |
|:-----|:-----|
| 🔥 2025/06 | Our survey paper is released on arXiv! |
| 🔥 2025/06 | Repository created! |

---

## 🔍 Table of Contents

- [📖 Introduction](#-introduction)
- [🏗️ Taxonomy](#️-taxonomy)
- [💬 Instruction Understanding](#-instruction-understanding)
  - [Privacy-Preserving Input](#privacy-preserving-input)
  - [Instruction-Side Safety](#instruction-side-safety)
  - [Instruction Grounding Accuracy](#instruction-grounding-accuracy)
  - [Instruction Fulfillment Utility](#instruction-fulfillment-utility)
  - [Instruction Parsing Efficiency](#instruction-parsing-efficiency)
  - [Robust Instruction Understanding](#robust-instruction-understanding)
  - [Instruction Generalization](#instruction-generalization)
  - [Human-Centered Instruction Understanding](#human-centered-instruction-understanding)
- [👁️ Environment Perception](#️-environment-perception)
  - [Privacy-Preserving Perception](#privacy-preserving-perception)
  - [Adversarial Perception Defense](#adversarial-perception-defense)
- [🔮 Future Directions](#-future-directions)
- [🔖 Citation](#-citation)
- [📧 Contact](#-contact)

---

## 📖 Introduction

<div align="center">
  <img src="assets/overview.png" width="90%" alt="Survey Overview"/>
  <br>
  <em>Fig. 1: Overview of our survey on Privacy-Aware Embodied AI.</em>
</div>

<br>

This paper reviews the current landscape of embodied AI methodologies with a focus on **privacy considerations**. We identify critical gaps in existing Vision-Language-Action (VLA) systems and propose future research directions for developing **privacy-aware**, **efficient**, and **trustworthy** embodied AI systems through a comprehensive closed-loop framework.

Key privacy concerns in embodied AI include:

- 🏠 **Environmental Privacy**: Home layouts, personal belongings, sensitive documents
- 👤 **Personal Privacy**: Human faces, activities, behavioral patterns  
- 🔐 **Data Security**: Training data leakage, model inversion attacks
- 📡 **Communication Privacy**: Data transmission vulnerabilities

---

## 🏗️ Taxonomy

<div align="center">
  <img src="assets/taxonomy.png" width="95%" alt="Taxonomy"/>
  <br>
  <em>Fig. 2: Taxonomy of Privacy-Aware Embodied AI. We systematically categorize the research into two core pillars: (1) Instruction Understanding, covering privacy-preserving input, safety, accuracy, utility, efficiency, robustness, generalization, and human-centered aspects; (2) Environment Perception, encompassing privacy-preserving perception and adversarial defense.</em>
</div>

---

## 💬 Instruction Understanding

### Privacy-Preserving Input

<div align="center">
  <img src="assets/privacy_input.png" width="80%" alt="Privacy-Preserving Input"/>
  <br>
  <em>Fig. 3: Privacy-preserving input strategies for embodied agents.</em>
</div>

<br>

Privacy-preserving input is critical for embodied agents that see homes, bodies and workflows, because leaks at the instruction stage directly undermine trust.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts](https://arxiv.org/abs/2508.02190) | - | - |
| 2025 | arXiv | [ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting](https://arxiv.org/abs/2502.14780) | - | - |
| 2024 | ICLR | [Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory](https://proceedings.iclr.cc/paper_files/paper/2024/file/08305d8b2ddab98932c163ea73df065f-Paper-Conference.pdf) | - | - |
| 2022 | ECCV | [FedVLN: Privacy-Preserving Federated Vision-and-Language Navigation](https://arxiv.org/abs/2203.14936) | - | - |
| 2021 | CIKM | [Natural Language Understanding with Privacy-Preserving BERT](https://doi.org/10.1145/3459637.3482281) | - | - |

### Instruction-Side Safety

Instruction-side safety addresses how unsafe prompts can trigger dangerous physical actions.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [AGENTSAFE: Benchmarking the Safety of Embodied Agents on Hazardous Instructions](https://arxiv.org/abs/2506.14697) | - | - |
| 2025 | arXiv | [Advancing Embodied Agent Security: From Safety Benchmarks to Input Moderation](https://arxiv.org/abs/2504.15699) | - | - |
| 2025 | arXiv | [BadNAVer: Exploring Jailbreak Attacks On Vision-and-Language Navigation](https://arxiv.org/abs/2505.12443) | - | - |
| 2025 | arXiv | [POEX: Towards Policy Executable Jailbreak Attacks Against the LLM-based Robots](https://arxiv.org/abs/2412.16633) | - | - |
| 2024 | arXiv | [SafeEmbodAI: a Safety Framework for Mobile Robots in Embodied AI Systems](https://arxiv.org/abs/2409.01630) | - | - |
| 2024 | ISSREW | [A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems](https://doi.org/10.1109/ISSREW63542.2024.00103) | - | - |

### Instruction Grounding Accuracy

Accuracy in instruction understanding is central because small misinterpretations of long, compositional commands can accumulate into large trajectory errors or task failures.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | Applied Intelligence | [A multilevel attention network with sub-instructions for continuous vision-and-language navigation](http://dx.doi.org/10.1007/s10489-025-06544-9) | - | - |
| 2024 | PAA | [Instruction-aligned hierarchical waypoint planner for vision-and-language navigation](https://api.semanticscholar.org/CorpusID:273112653) | - | - |
| 2024 | IROS | [Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation](https://doi.org/10.1109/IROS58592.2024.10801822) | - | - |
| 2024 | RA-L | [Learning Multimodal Confidence for Intention Recognition in Human-Robot Interaction](https://doi.org/10.1109/LRA.2024.3432352) | - | - |
| 2023 | ROBIO | [A Modular Framework for Robot Embodied Instruction Following by Large Language Model](https://doi.org/10.1109/ROBIO58561.2023.10355013) | - | - |
| 2021 | arXiv | [Learning a natural-language to LTL executable semantic parser for grounded robotics](https://arxiv.org/abs/2008.03277) | - | - |
| 2019 | ACL | [Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation](https://aclanthology.org/P19-1181/) | - | - |
| 2018 | Autonomous Robots | [Grounding natural language instructions to semantic goal representations](https://api.semanticscholar.org/CorpusID:41441732) | - | - |

### Instruction Fulfillment Utility

Instruction utility concerns how well an agent resolves semantic ambiguity to maximize success rate, path efficiency and overall usefulness.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [JARVIS: A Neuro-Symbolic Commonsense Reasoning Framework for Conversational Embodied Agents](https://arxiv.org/abs/2208.13266) | - | - |
| 2025 | arXiv | [REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](https://arxiv.org/abs/2505.10872) | - | - |
| 2025 | arXiv | [A Model-Agnostic Approach for Semantically Driven Disambiguation in Human-Robot Interaction](https://arxiv.org/abs/2409.17004) | - | - |
| 2023 | arXiv | [Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models](https://arxiv.org/abs/2310.15127) | - | - |
| 2022 | RA-L | [DoRO: Disambiguation of Referred Object for Embodied Agents](https://doi.org/10.1109/LRA.2022.3195198) | - | - |

### Instruction Parsing Efficiency

Instruction parsing efficiency asks how quickly and cheaply an agent can turn language into effective decisions under limited compute, data and interaction budgets.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers](https://arxiv.org/abs/2503.20384) | - | - |
| 2025 | RA-L | [Boosting Efficient Reinforcement Learning for Vision-and-Language Navigation With Open-Sourced LLM](https://doi.org/10.1109/LRA.2024.3511402) | - | - |
| 2024 | arXiv | [MAGIC: Meta-Ability Guided Interactive Chain-of-Distillation for Effective-and-Efficient VLN](https://arxiv.org/abs/2406.17960) | - | - |
| 2023 | arXiv | [Grounding Language with Visual Affordances over Unstructured Data](https://arxiv.org/abs/2210.01911) | - | - |
| 2022 | RA-L | [COSM2IC: Optimizing Real-Time Multi-Modal Instruction Comprehension](https://doi.org/10.1109/LRA.2022.3194683) | - | - |

### Robust Instruction Understanding

Robust instruction understanding emphasizes reliable task execution under noisy, incomplete or adversarial conditions.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [NavA³: Understanding Any Instruction, Navigating Anywhere, Finding Anything](https://arxiv.org/abs/2508.04598) | - | - |
| 2025 | arXiv | [Pragmatic Embodied Spoken Instruction Following in Human-Robot Collaboration with Theory of Mind](https://arxiv.org/abs/2409.10849) | - | - |
| 2025 | arXiv | [Learning Efficient and Robust Language-conditioned Manipulation using Textual-Visual Relevancy](https://arxiv.org/abs/2406.15677) | - | - |
| 2025 | CoRL Workshop | [Task Robustness via Re-Labelling Vision-Action Robot Data](https://openreview.net/forum?id=M6M5W0lmaY) | - | - |
| 2024 | arXiv | [Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models](https://arxiv.org/abs/2405.19802) | - | - |
| 2020 | IJRR | [Multimodal estimation and communication of latent semantic knowledge for robust execution](https://doi.org/10.1177/0278364920917755) | - | - |

### Instruction Generalization

Instruction-side generalization asks whether an agent can correctly follow novel, compositional commands about unseen objects, goals and environments.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | ACM MM | [OpenMap: Instruction Grounding via Open-Vocabulary Visual-Language Mapping](http://dx.doi.org/10.1145/3746027.3754887) | - | - |
| 2023 | arXiv | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) | - | - |
| 2023 | arXiv | [Instruction-Following Agents with Multimodal Transformer](https://arxiv.org/abs/2210.13431) | - | - |
| 2023 | arXiv | [ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings](https://arxiv.org/abs/2206.12403) | - | - |

### Human-Centered Instruction Understanding

Human-centered instruction understanding stresses that agents should reason about users' intentions, context, safety needs and mental models.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | Book Chapter | [Human-Centred Robotics and AI for Trustworthy Human-Robot Interaction](https://doi.org/10.1007/978-3-031-97673-5_9) | - | - |
| 2025 | arXiv | [Mixed-Initiative Dialog for Human-Robot Collaborative Manipulation](https://arxiv.org/abs/2508.05535) | - | - |
| 2025 | arXiv | [Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning](https://arxiv.org/abs/2504.00907) | - | - |
| 2024 | HRI | [Understanding Large-Language Model (LLM)-powered Human-Robot Interaction](http://dx.doi.org/10.1145/3610977.3634966) | - | - |
| 2023 | arXiv | [Human-Centric Autonomous Systems With LLMs for User Command Reasoning](https://arxiv.org/abs/2311.08206) | - | - |
| 2023 | SIGIR | [Evaluating Task-oriented Dialogue Systems with Users](https://doi.org/10.1145/3539618.3591788) | - | - |
| 2022 | RA-L | [DialFRED: Dialogue-Enabled Agents for Embodied Instruction Following](http://dx.doi.org/10.1109/LRA.2022.3193254) | - | - |
| 2022 | arXiv | [Dialog Acts for Task-Driven Embodied Agents](https://arxiv.org/abs/2209.12953) | - | - |
| 2018 | RoboDIAL | [Jointly Improving Parsing and Perception for Natural Language Commands through Human-Robot Dialog](http://nn.cs.utexas.edu/pub-view.php?PubID=127720) | - | - |

---

## 👁️ Environment Perception

### Privacy-Preserving Perception

<div align="center">
  <img src="assets/privacy_perception.png" width="80%" alt="Privacy-Preserving Perception"/>
  <br>
  <em>Fig. 4: Privacy-preserving perception strategies for embodied agents.</em>
</div>

<br>

As embodied agents enter homes and other sensitive spaces, environment perception becomes a major privacy risk.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [Improved Semantic Segmentation from Ultra-Low-Resolution RGB Images Applied to Privacy-Preserving Object-Goal Navigation](https://arxiv.org/abs/2507.16034) | - | - |
| 2025 | arXiv | [Real-Time Privacy Preservation for Robot Visual Perception](https://arxiv.org/abs/2505.05519) | - | - |
| 2025 | arXiv | [Privacy Risks of Robot Vision: A User Study on Image Modalities and Resolution](https://arxiv.org/abs/2505.07766) | - | - |
| 2025 | Applied Sciences | [Privacy-Preserved Visual Simultaneous Localization and Mapping Based on a Dual-Component Approach](https://www.mdpi.com/2076-3417/15/5/2583) | - | - |
| 2024 | J. Responsible Technology | [Inherently privacy-preserving vision for trustworthy autonomous systems: Needs and solutions](https://doi.org/10.1016/j.jrt.2024.100079) | - | - |
| 2024 | arXiv | [DT-RaDaR: Digital Twin Assisted Robot Navigation using Differential Ray-Tracing](https://arxiv.org/abs/2411.12284) | - | - |
| 2019 | IROS | [Privacy-Preserving Robot Vision with Anonymized Faces by Extreme Low Resolution](https://doi.org/10.1109/IROS40897.2019.8967681) | - | - |

### Adversarial Perception Defense

At the perception stage, safety is largely determined by adversarial robustness, since small perturbations to sensor inputs can cause severe failures.

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [Adversarial Attacks and Detection in Visual Place Recognition for Safer Robot Navigation](https://arxiv.org/abs/2506.15988) | - | - |
| 2025 | arXiv | [BadDepth: Backdoor Attacks Against Monocular Depth Estimation in the Physical World](https://arxiv.org/abs/2505.16154) | - | - |
| 2024 | arXiv | [Hijacking Vision-and-Language Navigation Agents with Adversarial Environmental Attacks](https://arxiv.org/abs/2412.02795) | - | - |
| 2024 | arXiv | [Malicious Path Manipulations via Exploitation of Representation Vulnerabilities of VLN Systems](https://arxiv.org/abs/2407.07392) | - | - |
| 2024 | arXiv | [Embodied Active Defense: Leveraging Recurrent Feedback to Counter Adversarial Patches](https://arxiv.org/abs/2404.00540) | - | - |

---

## 🔮 Future Directions

We identify several promising research directions:

### 🛡️ Privacy-Aware Protection

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Privacy-Aware VLA** | Integrating privacy-aware semantics with embodied performance; user-defined "red-line rules" | [OpenVLA](https://arxiv.org/abs/2406.09246), [RT-2](https://arxiv.org/abs/2307.15818) |
| **Federated Embodied AI** | "Data stays, model moves" paradigm; knowledge aggregation across heterogeneous robots | [FedVLN](https://arxiv.org/abs/2203.14936), [FedVLA](https://arxiv.org/abs/2508.02190) |
| **On-Device Inference** | Edge intelligence with embodiment-aware efficiency mechanisms | [TinyVLA](https://arxiv.org/abs/2406.04339), [BitVLA](https://arxiv.org/abs/2506.07530) |

### ⚡ Inference Acceleration and Efficiency

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Persistent Computation Caching** | Cross-episode reusable computations for repetitive tasks | [VLA-Cache](https://arxiv.org/abs/2502.02175) |
| **Asynchronous Multi-Level Inference** | Parallel execution of slow semantic reasoning and fast reactive control | [OpenHelix](https://arxiv.org/abs/2505.03912), [DEER-VLA](https://arxiv.org/abs/2410.13383) |

### ⚖️ Utility-Accuracy-Privacy Triangle

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Multi-Objective Optimization** | Pareto frontiers across task performance, computational cost, and privacy guarantees | [Efficiency Survey](https://arxiv.org/abs/2510.24795) |
| **Privacy-Constrained Data Efficiency** | Data-efficient learning paradigms under privacy budgets | [Learning with Impartiality](https://arxiv.org/abs/2302.09183) |
| **Context-Aware Adaptive Mechanisms** | Runtime context-aware frameworks for adaptive privacy protection | - |

### 🦺 Safety-Aware Protection

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Safety-Constrained VLA** | CMDP-based constrained learning; automatic safety predicate discovery | [SafeVLA](https://arxiv.org/abs/2310.12773), [Safe-RLHF](https://arxiv.org/abs/2310.12773) |

### 🎯 Generalization & Robustness

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Hierarchical Generalization** | Unified framework for within-task and cross-task generalization | [RL-VLA Study](https://arxiv.org/abs/2505.15660) |
| **Multi-Level Perturbation Resilience** | Integrated evaluation-defense frameworks | [Eva-VLA](https://arxiv.org/abs/2509.18953) |

### 👤 Human-Centric Alignment

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Human Preference-Aligned VLA** | Multi-dimensional preference optimization; personalized VLA policies | [GRAPE](https://arxiv.org/abs/2411.19309), [FLaRe](https://arxiv.org/abs/2409.16578) |

---

## 🔖 Citation

If you find this survey helpful, please consider citing:

```bibtex
@misc{author2025privacyaware,
  title={Privacy-Aware Embodied AI: Towards a Comprehensive Framework},
  author={Author Name and Co-author Name},
  year={2025},
  eprint={2510.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2510.xxxxx},
}
```

---

## 📧 Contact

For any questions or suggestions, please feel free to contact us:

<div align="center">

| Author | Email | Affiliation |
|:-------|:------|:------------|
| **Your Name** | your.email@example.com | Your University |

</div>

---

## 🙏 Acknowledgements

We thank all the authors of the papers included in this survey. Special thanks to:
- [A Survey on Efficient Vision-Language-Action Models](https://arxiv.org/abs/2510.24795)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)

---

<div align="center">

**If you find this repository useful, please give us a ⭐!**

</div>

---

<p align="center">
  <i>Last updated: June 2025</i>
</p>
