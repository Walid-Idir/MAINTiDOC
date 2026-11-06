# Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel

|LayerType|ComplexityperLayer|Sequential<br>Operations|MaximumPathLength|
|---|---|---|---|
|Self-Attention<br>Recurrent<br>Convolutional<br>Self-Attention (restricted)|_O_(_n_2 _· d_)<br>_O_(_n · d_2)<br>_O_(_k · n · d_2)<br>_O_(_r · n · d_)|_O_(1)<br>_O_(_n_)<br>_O_(1)<br>_O_(1)|_O_(1)<br>_O_(_n_)<br>_O_(_logk_(_n_))<br>_O_(_n/r_)|

