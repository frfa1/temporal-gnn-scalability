# Evaluating the scalibility of graph neural networks on regional forecasting of Danish wind power

This project is a Master Thesis, which aims at evaluating how GNN architectures scale when applied to different numbers of wind turbines from real-world Danish wind power data. The models are compared with two naive baselines: A historical mean and a persistence model. Further, the models are compared with a Gated Recurrent Unit (GRU) Encoder-Decoder model. The GRU model takes as input historical weather data and turbine outputs in the encoder, and forecasted weather data in the decoder, to produce forecasted turbine outputs.

The GNN architectures add to the GRU model by providing a graph convolutional layer to the forecasted weather data, before it is fed into the decoder.

Much of the code it designed for the specific data, but the model architectures may be of use to others.

## Repository structure

- **analysis**: Provides tool for analysis and visualization of the data and results of experiments
- **evaluation**: Evaluation metrics and tools to fetch results from trained and tested models
- **loader**: Tools for loading the data as a PyTorch DataLoader and PyTorch Geometric Temporal Loader
- **models**: Different model architectures
- **preprocessing**: Initial preprocessing of large scale data and feature processing tools for training models
- **train_evaluate**: Tools for data splitting, running hyperparameter-tuning and doing the final training and testing of the models
