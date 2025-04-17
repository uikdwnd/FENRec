# FENRec

This is our Pytorch implementation for the paper: Future Sight and Tough Fights: Revolutionizing Sequential Recommendation with FENRec

## Dataset

In our experiments, we utilize 4 datasets, all stored in the `./data` folder.
- For the Beauty, Sports, Toys, and Yelp datasets, we employed the datasets downloaded from [this repository](https://github.com/salesforce/ICLRec). 

## Environment
Please run the following command to create the Conda environment to run our code:
```
conda env create -f env.yml
```
After creating the Conda environment, you can activate the environment:
```
conda activate FENRec
```

## Train Model:

After activating the Conda environment, you can use the training scripts in the `./src/scrips` folder to train the model:
```
bash beauty.sh
bash sports.sh
bash toys.sh
bash yelp.sh
```