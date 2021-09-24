# ESAM: Discriminative Domain Adaptation with Non-Displayed Items to Improve Long-Tail Performance
- Authors: Zhihong Chen, Rong Xiao, Chenliang Li, Gangfeng Ye, Haochuan Sun, Hongbo Deng
- Journal/Conference: SIGIR
- Year: 2020

# Summary
## Innovation
This paper proposes an entire space adaptation model (ESAM) to address the long-tail problem from the perspective of domain adaptation (DA), without requiring any auxiliary information and auxiliary domains.Two effective regularization strategies, i.e., center-wise clustering and self-training are introduced to improve DA process.

## Methods
This paper is based on the Domain Adaptation in transfer learning. The difference between covariances of source item matrix and target item matrix is minimized to decrease the distance between distributions of source domain and target domain. Two regularization strategies are introduced to improve the DA process.
- For source clustering, center-wise clustering aims to reduce intra-class discrepancy and enforce inter-class separability.
- For target clustering, self-training aims to increse target discrimination and avoid negative transfer with pseudo-label.
The model is trained through gradient descent with these metrics combined as loss function.

## Problem
- How to assign those pseudo-labels to the target items?

