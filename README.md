# <img src="images/logo.png" height="100" alt="logo">

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

One very efficient coding technique when dealing with large arrays containing
binary values is the so called multispin-coding. Instead of using numbers or
boolean values to represent each individual spin, multiple spins are stored
inside of one computer word. Using logical binary operators one can evaluate
multiple spins at once making this method very time efficient.
