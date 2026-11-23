
# conformalRNN

<!-- badges: start -->
<!-- badges: end -->

conformalRNN provides tools for multi-horizon time series forecasting with conformal prediction. 
The package builds on split conformal prediction, a distribution-free framework that 
constructs prediction intervals with finite-sample marginal coverage of at least (1-\alpha\) 
under exchangeability of the calibration and future observations. 
Because this guarantee depends on exchangeability rather than a specific parametric distribution, 
careful consideration of the data structure is important. 
In particular, the method is intended for settings with multiple independent time series 
where the calibration and future series can reasonably be viewed as arising from the same underlying data-generating mechanism 
and therefore exhibit comparable statistical characteristics.


## Installation

You can install the development version of conformalRNN from [GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("SayheeK/conformalRNN")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(conformalRNN)
## basic example code
```

