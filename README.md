This is our implementation based on CardioGAN: Attentive Generative Adversarial Network with Dual Discriminators for Synthesis of ECG from PPG
https://github.com/pritamqu/ppg2ecg-cardiogan.git

We rebuild layers and module under Pytorch, and add discriminators in both time and frequency domain.
We use dataset from: 
https://physionet.org/content/bidmc/1.0.0/

Weights can be used on the generator: https://github.com/pritamqu/ppg2ecg-cardiogan/releases/download/model_weights/cardiogan_ppg2ecg_generator.zip

And try test_cardiogan.py to test the model.
