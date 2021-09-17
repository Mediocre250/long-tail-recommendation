# On both Cold-Start and Long-Tail Recommendation with Social Data
- Authors: Jingjing Li, Ke Lu, Zi Huang and Heng Tao Shen
- Journal/Conference: TKDE
- Year: 2021

# Summary
## Innovation
This paper is the first one aiming to challenge the cold-start problem and long-tail problem simutaneously.
- For the cold-start problem, applying the ideas in transfer learning, this paper transfers the knowledge learned from side information( like user profile or social relationship) into the target system.
- For the long-tail problem, the items are decomposed into two parts, low-rank part( short-head items) and sparse part( long-tail items). Then these two parts are trained independently and merged into the final recommendation.
## Methods
Considering both the popular items and the niche items, the user-item matrix is split into two parts, one for short-head and the other for long-tail.  
Using a linear model, this paper factorizes each user-item matrix into the side information matrix and a mapping matrix which maps the user relationship to the user-item interaction.  
Penalty function is added to ensure that the short-head is low-rank and the long-tail is sparse.  
## Problem
- Why is it safe to assume that users preferences and their social/interest groups can be formulated as a linear regression problem?


