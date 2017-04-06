# Vision-Based-Tracking-Course-Project

## Project 1 Particle Filter
1. Use the ground truth bounding box as the observations, add observation noises and false positives.
2. Modified the PF tracker to use foreground/background binary masks to compute observation likelihoods, which resulted in a much more practical tracker could be easily used on other video data taken from a stationary camera. The method to compute the likelihood is to keep around the width and height information about the bounding box of the object that is tracked. Then, to evaluate a sample at location (x,y), examine the rectangular region centered at (x,y) and count how many pixels are on (foreground pixels) in that box vs off (background pixels). We would expect the likelihood should be higher if there is a higher proportion of foreground pixels.
