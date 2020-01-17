# Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/7Kk_RsGycHs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Questions

## Drift

![drift](drift.png)

Q. Why is it not possible to detect the signal in this example?

A. Because the noise (drift) is "louder" than the signal itself.

## Aliasing

![hrf](hrf.png)

Q. Given a TR of 2000ms, is it possible to avoid aliasing when sampling from the HRF shown above? Why or why not?

A. No. Signals faster than 1/2 the sampling rate—called the Nyquist frquency—will be aliased. To avoid aliasing, you must sample at least twice as fast as the fastest frequency in the signal. Here, the fastest frequency is ~2000ms, so the sampling rate is not at least twice as fast as the fastest frequency and therefore aliasing occurs.

Q. What could possibly be the largest TR while avoiding aliasing? Why?

A. Assuming "largest TR" means longest time for each sample, about 1000ms because that's about twice as fast as the fastest frequency (~2000ms).

## Noise Map

![noise map](noise-map.png)

Q. If the noise were uniform across the entire brain, what would the spatial map look like instead?

A. The whole thing would look uniformly noisy instead of like it is where we can identify brain structure.

Q. If noise were higher closer to the center of the brain, what would the spatial map look like instead?

A. We would be less able to make out the structure of the center of the brain because it would be noisy.
