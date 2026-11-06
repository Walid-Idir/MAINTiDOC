# Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

|Model|BLEU<br>EN DE EN FR<br>- -|Col3|TrainingCost(FLOPs)<br>EN DE EN FR<br>- -|
|---|---|---|---|
|ByteNet [15]|23.75|||
|Deep-Att + PosUnk [32]<br>GNMT + RL [31]<br>ConvS2S [8]<br>MoE[26]|24.6<br>25.16<br>26.03|39.2<br>39.92<br>40.46<br>40.56|1_._0_ ·_ 1020<br>2_._3_ ·_ 1019<br>1_._4_ ·_ 1020<br>9_._6_ ·_ 1018<br>1_._5_ ·_ 1020<br>2_._0_ ·_ 1019<br>1_._2_ ·_ 1020|
|Deep-Att + PosUnk Ensemble [32]<br>GNMT + RL Ensemble [31]<br>ConvS2S Ensemble[8]|26.30<br>26.36|40.4<br>41.16<br>**41.29**|8_._0_ ·_ 1020<br>1_._8_ ·_ 1020<br>1_._1_ ·_ 1021<br>7_._7_ ·_ 1019<br>1_._2_ ·_ 1021|
|Transformer (base model)<br>Transformer (big)|27.3<br>**28.4**|38.1<br>**41.0**|**3****_._3****_ ·_ 1018**<br>2_._3_ ·_ 1019|

