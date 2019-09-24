# Monte Carlo simulation library for CPU, Nvidia CUDA and Xlinix FPGA

# Library structure

*	Xoshiro128** Pseudo Random Number Generator (PRNG)
    -	Independent sequences
*	Distribution generators:
  -	32-bit random integer to real uniform transform 
  -	Inverse Cumulative Normal Transform
*	Sobol sequence Quasi Random Number Generator (QRNG)

  -	Owen’s random scrambling
  -	21201 dimensions
  -	Sequence jump ahead

*	Brownian Bridge transform
  -	arbitrary time grid
  
*	CPU, CUDA and Xilinx HLS support

*	Single and double precision support

*	Unit tests

The CPU and host code includes are under the root namespace `sci`, CUDA includes are
under `sci::cuda` namespace and Xilinx FPGA HLS includes are in `sci::hls`.

# Examples

* The number pi estimation [mcPi](../mcPi), [mcPiQ](../mcPiQ)
* Arithmetic Asian option [mcArithmAsian](../mcArithmAsian), [mcArithmAsianQ](../mcArithmAsianQ) 

# Tests
All tests except RNG statistical tests are located in the test directory. The tests require a minor setup for the Boost and QuantLib libraries and do not require pre-built binaries, only the source files. The `QUANTLIB_ROOT` and `BOOST_ROOT` environment variables must point to the QuantLib and Boost source directories. The test suite is using the Catch2 unit testing framework, and it picks its header file from the `libs` directory in the root of the repository.
The RNG statistical test is performed using the PractRand test suite. The `test_rng` project produces the binary stdout that can be consumed by PractRand executable. 
For example:

```
test_rng.exe | PractRand_094\bin\msvc12_64bit\RNG_test.exe stdin32
```

