# SSL-audio
A PyTorch implementation comparing Contrastive Loss vs Barlow Twins for SSL audio classification using a hybrid 1D/2D ResNet encoder.

Methods
Supervised: ResNet2D trained with cross‑entropy on log‑mel spectrograms.

Contrastive: Joint training of ResNet1D (wav) and ResNet2D (mel‑spectrogram) with Contrastive Loss.

Barlow Twins: Same architecture but with Barlow Twins loss.

Results (Validation Accuracy):

Supervised ResNet2D: 95%;
Contrastive Learning: 75%;
Barlow Twins: 80%.

t‑SNE visualisation of the embeddings shows that Barlow Twins produce well‑separated digit clusters, while contrastive learning yields more mixed clusters.




