# Local Projections in Python

This package implements local projections in Python. It is based on the R package lpirfs and Jorda (2005) paper.

## Installation
This package is not yet registered on PyPI. To install, clone the repository and run `pip install .` in the root directory.

## Usage
Basic usage is a 3-step process:
 - Create a `LP` object specifying the model you'd like to estimate
 - Call the `estimate` method on the object to estimate impulse responses for a given horizon and shock size
 - Plot the results using the `plot_irfs` function

 The `LP` constructor is where you configure your specification. You can specify endogenous variables/controls with varying lag lengths, response variables, exogenous/identified shocks, interaction terms, trends. If you pass in a panel dataset, a panel LP will be estimated.

 See the documentation for `LP`, `estimate`, and `plot_irfs` for more details.