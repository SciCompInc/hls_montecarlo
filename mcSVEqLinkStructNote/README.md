#  Equity Linked Note option pricing example

Equity Linked Notes (ELNs) allow clients to customize return to suit their
investment needs. Traditional equity investments provide full exposure to
the market, whether the performance is positive or negative. ELNs provide
clients with an alternative to traditional equities that can offer an
enhanced return if the underlying equity asset rises, and varying levels
of protection if the market falls. ELNs can be linked to a variety of 
underlying assets, including indices, single stocks, portfolios of shares
and industry sectors, and can be expanded to include commodities and 
currencies.

The mcSVEqLinkStructNote example prices an equity-linked structured note
in which the note is linked to a basket of indices. If on observation dates
the performance of all indices is above the knockout barrier for that date, 
the note redeems early at par plus a bonus coupon. If early
redemption does not occur, then at maturity either:

1. the Note redeems at par plus a maturity coupon, or
2. if the performance of at least `KINum` of the indices have even been below
   the knock in barrier during the tenor, on a continuously observed basis,
   then the Note redeems at the performance percentage of the worst index.

The indices follow correlated [Heston](https://en.wikipedia.org/wiki/Heston_model)
stochastic volatility processes and we allow for a term structure of rates.

There are two versions of the code:

* [Standard Monte Carlo](.) (this directory)

* [Quasi Monte Carlo](../mcSVEqLinkStructNoteQ)

## FPGA Metrics

| Name            | BRAM_48K  | DSP48E  | FF  | LUT  |
|-----------------|:---------:|--------:|----:|-----:|
| Utilization (%) |  19       | 75      | 35  | 57   |

## CPU vs GPU vs FPGA benchmarking
OS: Ubuntu 16.04LTS
Development environments:
* CPU: Intel Parallel Studio XE 2019 
* GPU: CUDA 10.1, GCC 5.4.0
* FPGA: SDAccel 2018.3

| Hardware | Model                | MMpath/sec  | W   | paths/sec/mW  |
|----------|:--------------------:|------------:|----:|--------------:|
| CPU      | Intel Xeon E5-2686v4 | 0.427       | 110 | 4             |
| FPGA     | Xilinx Alveo U200    | 2.222       | 26  | 85            |
| GPU      | Nvidia V100 16GB     | 3.922       | 203 | 20            |

## Power estimation

* CPU: Intel SoC Watch
* GPU: NVIDIA SMI
* FPGA: Vivado Power Estimation



