[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/fbvV3TEg)
# Advanced Statistics Lab
## Jointly Gaussian Random Variables

This lab focuses on jointly Gaussian random variables.

Topics:
- Bivariate Gaussian joint PDF
- Marginal Gaussian PDFs
- Covariance matrix
- Correlation coefficient
- Independence for jointly Gaussian variables
- Simulation-based verification

---

## Background

A pair `(X, Y)` is jointly Gaussian if its joint PDF is:

    f_XY(x,y)
    =
    1 / (2*pi*sigma_X*sigma_Y*sqrt(1-rho^2))
    *
    exp( -Q(x,y) / (2*(1-rho^2)) )

where

    Q(x,y)
    =
    ((x - mu_X)^2 / sigma_X^2)
    - 2*rho*((x - mu_X)*(y - mu_Y))/(sigma_X*sigma_Y)
    + ((y - mu_Y)^2 / sigma_Y^2)

The parameters are:

    mu_X, mu_Y        = means
    sigma_X, sigma_Y = standard deviations
    rho              = correlation coefficient

For jointly Gaussian variables:

    rho = 0  implies  X and Y are independent

This is a special Gaussian property.

---

## Question 1 — Joint Gaussian PDF and Marginals

You are given:

    mu_X    = 1
    mu_Y    = -2
    sigma_X = 2
    sigma_Y = 3
    rho     = 0.6

Tasks:

1. Implement the joint Gaussian PDF.
2. Implement the marginal PDF of `X`.
3. Implement the marginal PDF of `Y`.
4. Implement the covariance matrix:

       [[sigma_X^2, rho*sigma_X*sigma_Y],
        [rho*sigma_X*sigma_Y, sigma_Y^2]]

5. Numerically estimate the double integral of the joint PDF over the rectangle:

       mu_X - 4*sigma_X <= x <= mu_X + 4*sigma_X

       mu_Y - 4*sigma_Y <= y <= mu_Y + 4*sigma_Y

---

## Question 2 — Independence and Simulation

Use the same Gaussian parameters.

Tasks:

1. Generate samples from a jointly Gaussian distribution.
2. Estimate sample means.
3. Estimate sample covariance matrix.
4. Estimate sample correlation coefficient.
5. Check whether the pair is independent when `rho = 0`.
6. Check whether the pair is independent when `rho` is nonzero.
7. Verify numerically that `rho = 0` gives approximately zero sample covariance.
8. Verify numerically that nonzero `rho` gives nonzero sample covariance.

---

## Rules

Students must implement all functions in:

    AI_stats_lab.py

Do not modify the test file.

---

## Run Locally

Install:

    pip install numpy pytest

Run:

    pytest
