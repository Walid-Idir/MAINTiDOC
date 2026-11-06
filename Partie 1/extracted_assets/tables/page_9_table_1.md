# Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed

|Col1|train<br>N d model d ff h d k d v P drop ϵls steps|PPL BLEU params<br>(dev) (dev) ×106|
|---|---|---|
|base|6<br>512<br>2048<br>8<br>64<br>64<br>0.1<br>0.1<br>100K|4.92<br>25.8<br>65|
|(A)|1<br>512<br>512<br>4<br>128<br>128<br>16<br>32<br>32<br>32<br>16<br>16|5.29<br>24.9<br>5.00<br>25.5<br>4.91<br>25.8<br>5.01<br>25.4|
|(B)|16<br>32|5.16<br>25.1<br>58<br>5.01<br>25.4<br>60|
|(C)|2<br>4<br>8<br>256<br>32<br>32<br>1024<br>128<br>128<br>1024<br>4096|6.11<br>23.7<br>36<br>5.19<br>25.3<br>50<br>4.88<br>25.5<br>80<br>5.75<br>24.5<br>28<br>4.66<br>26.0<br>168<br>5.12<br>25.4<br>53<br>4.75<br>26.2<br>90|
|(D)|0.0<br>0.2<br>0.0<br>0.2|5.77<br>24.6<br>4.95<br>25.5<br>4.67<br>25.3<br>5.47<br>25.7|
|(E)|positional embedding instead of sinusoids|4.92<br>25.7|
|big|6<br>1024<br>4096<br>16<br>0.3<br>300K|**4.33**<br>**26.4**<br>213|

