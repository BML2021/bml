### BML

#### Framework

* GCAN-train.py

    Train model GCAN and then test the model on the test-set.

* abml-GCAN.py

    Implement of training and testing on our proposed model BML

* abml_GCAN_without_z.py

    Ablation study of our BML model.

* GCAN.py

    Define the layers in model GCAN.

* topics_embedding.py

    Generate embedding of topics.

* dataset.py

    Preprocess data from the original dataset.

* utils.py

    Implement of some auxiliary functions.

#### Running our code

```sh
python abml_GCAN.py --data_scale=1.0 --meta_lr=0.0005 --inner_lr=0.0005 --retweet_user_size=25 --resume_epoch=0 --num_epochs=200 --log_file='log_meta_25_10.txt'
python GCAN-train.py --retweet_user_size=25 --data_scale=1.0 --log_file='log_gcan_25_10.txt'
python abml_GCAN_without_z.py --meta_lr=0.0005 --inner_lr=0.001 --retweet_user_size=25 --resume_epoch=0 --num_epochs=300 --log_file='log_non_z_25_10.txt'
```

