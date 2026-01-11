Bias: Bias is the inability for a mcahine learning method (like linear regression) to capture the true relationship
![alt text](image-1.png)


If we do overfitting, we get a very low MSE for the training set, but a very high one for the Test set. 
![alt text](image.png)

This leads to a high Variance - this is the case in overfit

In ML we want low bias and low variability (high consistency)
This happens by finding the sweet spot between simple (high bias) and complicated (high variance) models

3 commonly used methods are:

1. regularization
2. boosting
3. bagging


Normalizing values Turns numbers into z-scores so they have mean 0 and variance 1.
This is common preprocessing in ML.

# Day 1 Notes

## What I built
- normalize(): z-score scaling with safe std=0 handling
- train_test_split_indices(): reproducible shuffle split using seed
- rmse(): metric for regression error

## What I understood (in my words)
- Seed means:
- Why we clamp test_size:
- Why we test code:

## One thing I struggled with
- 

## One question for the team on Day 1 at Proxy Foods
- What split strategy do we use to avoid leakage? (e.g., by recipe_id, by panelist_id, time-based?)
