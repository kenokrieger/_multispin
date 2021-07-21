# <img src="images/logo.png" height="100" alt="logo">

<img src="images/license.svg" alt="MIT License">

## Outline

This particular model attempts to predict the behavior of traders in a market
governed by two simple guidelines:

- Do as neighbors do

- Do what the minority does

mathematically speaking is each trader represented by a spin on a three dimensional
grid. The local field of each spin *S*<sub>i</sub> is given by the equation below

<img src="images/local_field.png" alt="field equation" height="100">

where *J*<sub>ij</sub> = *j* for the nearest neighbors and 0 otherwise. The spins
are updated according to a Heatbath dynamic which reads as follows

<img src="images/spin_updates.png" alt="Heatbath equation" height="100">


The model is thus controlled by the three parameters

- &alpha;, which represents the tendency of the traders to be in the minority

- *j*, which affects how likely it is for a trader to pick up the strategy of its neighbor

- &beta;, which controls the randomness

(For more details see <a href="https://arxiv.org/pdf/cond-mat/0105224.pdf">
S.Bornholdt, "Expectation bubbles in a spin model of markets: Intermittency from
frustration across scales, 2001"</a>)

## Implementation

### 3D Metropolis Algorithm

The main idea behind the metropolis algorithm is to split the main lattice into
two sub-lattices with half of the original grid width. You can think of these lattices
as tiles on a chessboard (see figure below).</br>
<img src="images/metropolis3d.png" alt="3d metropolis algorithm" height="350"> </br>
Each black or white tile at position p = (row, col, lattice_id) can be assigned
an individual index in a 1 dimensional array.


### Precomputation

Looking at the equation from the outline one can see, that for each iteration
there exist 14 possible values for the probability *p*. These values can be
precomputed and assigned an index ranging from 0 to 13.

```c++
void compute_probabilities(float* probabilities, const float market_coupling, const float reduced_j)
{
    for (int idx = 0; idx < 14; idx++) {
        double field = reduced_j * (2 * idx - 6 - 14 * (idx / 7)) + market_coupling * ((idx < 7) ? -1 : 1);
        probabilities[idx] = 1 / (1 + exp(field));
    }
}
```

Instead of computing the probability for each spin, the kernel now only has to
find the respective value for each individual spin in the array which only depends
on the sum over the 6 neighbors and its own orientation.

```c++
float probability = probabilities[7 * ((traders[index] < 0) ? 0 : 1) + (neighbor_sum + 6) / 2];
traders[index] = random_values[index] < probability ? 1 : -1;
```

### Multispin Coding Approach

Multispin coding stores spin values in individual bits rather than full bytes leading to more efficient memory usage and thus faster computation times.
The spin values are mapped from (-1, 1) to the binary tuple (0, 1). This
allows for each spin to be resembled by an individual bit. An unsigned long
long (with size of 64 bits), for example, can store 16 spins. The remaining
bits are left untouched to enable a fast computation of the nearest neighbors
sum. Lets look at a simplified example of storing two spins in a 8 bit
variable:

The tuple of source spins is represented by 00010001 with nearest neighbors
00000000, 00010000, 00000000, 00000001. In this example we are dealing with two spins in parallel. To compute the nearest neighbors sum it is sufficient
to simply add all four values and look at the resulting bits.

- 00000000 =  0
- 00010000 = 16
- 00000000 = 0
- 00000001 = 1

The sum over all neighbors thus equals 17.

- 17 = 00010001

To find the number of neighbors with spin up for each individual spin, one
only needs to look at the 4 bits allocated for the storage of the spin. In
this example both spins have one neighbor with spin up.

## Compiling

### On Ubuntu/Linux

To compile and run the executable you need to have a system with a CUDA-capable
GPU. Additionaly you need to install the cuda-toolkit as well as suitable
C/C++ compiler (e.g. gcc). If everything is set up correctly you should be able
produce the executable with the `make` command.

**Note:** You may need to adjust the `-arch` option in the Makefile according
to the compute capabilities of your GPU.
