import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch
from model.LSTM import LSTMModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:
    def __init__(self, stock, train_size, data, 
                        col, model, weight_path = None):
        
        self.stock = stock
        self.origin_data = data
        self.data = data.iloc[:, col: col + 1]
        self.train_size = train_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
           

        if weight_path:
            self.model.load_state_dict(torch.load(weight_path))

        self.scaled_dataset()

    def scaled_dataset(self):
        self.scaler  = MinMaxScaler(feature_range = (0, 1))
        self.scaled_dataset = self.scaler.fit_transform(self.data)

    def create_dataset(self, n_lookback = 60,
                            n_predict = 30):
        

        self.n_lookback = n_lookback
        self.n_predict = n_predict

        X, y = [], []
        for i in range(n_lookback, len(self.scaled_dataset) - n_predict + 1):
            X.append(self.scaled_dataset[i - n_lookback : i])
            y.append(self.scaled_dataset[i: i + n_predict])
        X, y = torch.Tensor(X), torch.Tensor(y)

        # Convert data to PyTorch tensors
        X_train, y_train = X[ : self.train_size], y[: self.train_size]
        X_test, y_test = X[self.train_size : ], y[self.train_size: ]

        self.model = self.model(1, 128, 2, n_predict, self.device).to(self.device) 
        return X_train, y_train, X_test, y_test

    def dataloader(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.create_dataset()

        batch_size = 32

        
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size = batch_size)

        test_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size = batch_size)
        
        print(len(train_loader))
        print(len(test_loader))
        return train_loader, test_loader
    

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print('Saving the model')


    def train_model(self, num_epochs, model_path):

        self.train_hist = []
        self.test_hist = []
        self.num_epochs = num_epochs

        train_loader, test_loader = self.dataloader()
        total_step = len(train_loader)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 5e-4)

        for epoch in range(num_epochs):
            total_loss = 0.0

            pbar = tqdm(enumerate(train_loader), total = total_step,
                        desc = f"Epoch {epoch + 1}/{num_epochs}",
                        unit = "batch")

            # Training
            self.model.train()
            for i, (batch_X, batch_y) in pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                # print(predictions.shape)
                loss = loss_fn(predictions, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                pbar.set_postfix({"Loss": loss.item()})

            # Calculate average training loss and accuracy
            average_loss = total_loss / len(train_loader)
            self.train_hist.append(average_loss)

            # Validation on test data
            self.model.eval()
            with torch.no_grad():
                total_test_loss = 0.0

                for batch_X_test, batch_y_test in test_loader:
                    batch_X_test, batch_y_test = batch_X_test.to(self.device), batch_y_test.to(self.device)
                    predictions_test = self.model(batch_X_test)
                    test_loss = loss_fn(predictions_test, batch_y_test)

                    total_test_loss += test_loss.item()

                # Calculate average test loss and accuracy
                average_test_loss = total_test_loss / len(test_loader)
                self.test_hist.append(average_test_loss)

            print(f'Training Loss: {average_loss:.4f} - Test Loss: {average_test_loss:.4f}')


            self.save_model(model_path)

        self.plot_loss(f'./Images/Loss_LSTM_{self.stock}.png')
        self.plot_predict(f'{self.stock}_Predict', f'./Images/Plot_{self.stock}_Predict')

    def plot_loss(self, save_path):

        plt.figure(figsize = (7, 5))
        plt.plot(range(self.num_epochs), self.train_hist, 
                            label = "Training Loss")
        plt.plot(range(self.num_epochs), self.test_hist, 
                            label = "Testing Loss")
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value (MSE)")
        plt.legend()
        plt.title('Train vs Val Loss')
        plt.savefig(save_path)

        

    def plot_predict(self, name_plot, save_path):
        
        X_ = self.scaled_dataset[ -self.n_lookback : ]
        X_ = torch.Tensor(X_.reshape(1, self.n_lookback, 1)).to(self.device)

        with torch.no_grad():
            Y_ = self.model(X_)
            print(Y_.shape)
            Y_ = self.scaler.inverse_transform(Y_.squeeze(0).cpu().numpy())
        
        print(Y_)
        fig, ax = plt.subplots(figsize = (20, 4))

        # organize the results in a data frame
        df_past = self.origin_data[['Close']].reset_index()
        df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods = self.n_predict)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan

        results = pd.concat([df_past, df_future]).set_index('Date')
        results.to_csv(f'Predict_{self.n_predict}_{self.stock}.csv', index = True)
        # plot the results
        results.plot(title= name_plot, ax = ax)
        fig.savefig(save_path, bbox_inches = 'tight')

        plt.show()
