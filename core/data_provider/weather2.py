import numpy as np
import pandas as pd
import random

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.input_length = input_param['input_length']
        self.total_length = input_param['total_length']
        self.overlap = input_param.get('overlap', 0)  # Added overlap parameter
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.nan_fill_value = 0.0
        self.current_batch = None
        self.load()

    def load(self):
        weather = pd.read_csv(self.paths[0])
        if weather.isnull().values.any():
            print("Warning: NaN values found in data_file. Replacing with nan_fill_value.")
            weather.fillna(self.nan_fill_value, inplace=True)
        self.data['weather'] = weather.T.values

        if self.num_paths > 1:
            moran = pd.read_csv(self.paths[1])
            if moran.isnull().values.any():
                print("Warning: NaN values found in data_file. Replacing with nan_fill_value.")
                moran.fillna(self.nan_fill_value, inplace=True)
            self.data['moran'] = moran.T.values

    def total(self):
        return self.data['weather'].shape[0]

    def begin(self, do_shuffle=False):
        self.indices = np.arange(self.total(), dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_size = min(self.minibatch_size, self.total())
        self.update_batch_indices()

    def update_batch_indices(self):
        #print(self.total)
        start = max(0, self.current_position)
        end = min(self.total(), self.current_position + self.total_length)
        self.current_batch_indices = self.indices[start:end]
        #self.current_position += self.total_length - self.overlap
        # print(f"Updated batch indices: {self.current_batch_indices}")  # Debug
        # print(f"Current position: {self.current_position}, Total length: {self.total_length}, Overlap: {self.overlap}")  # Debug

    def next(self):
        self.current_position += self.total_length - self.overlap
        if self.no_batch_left():
            return None
        self.update_batch_indices()

    def no_batch_left(self):
       return self.current_position+self.total_length >= self.total()

    def input_batch(self):
        if self.no_batch_left():
            return None

        data_dim = self.data['weather'].shape
        num_inputs = len(self.paths)

        input_sequence = np.zeros((num_inputs, 1, self.input_length, data_dim[1])).astype(self.input_data_type)

        input_batch_indices = self.current_batch_indices[:self.input_length]

        for i, idx in enumerate(input_batch_indices):
            input_sequence[0, 0,i,:] = self.data['weather'][idx]
            if num_inputs > 1:
                input_sequence[1, 0,i,:] = self.data['moran'][idx]
        return input_sequence

    def output_batch(self):
        if self.no_batch_left():
            return None

        data_dim = self.data['weather'].shape
        num_inputs = len(self.paths)
        self.output_length = self.total_length - self.input_length
        output_sequence = np.zeros((num_inputs, 1, self.output_length, data_dim[1])).astype(self.output_data_type)

        #output_batch_indices = self.current_batch_indices + self.input_length
        output_batch_indices = self.current_batch_indices[self.input_length:]

        for i, idx in enumerate(output_batch_indices):
            if idx < self.total():
                output_sequence[0,0,i,:] = self.data['weather'][idx]
                if num_inputs > 1:
                    output_sequence[1, 0,i,:] = self.data['moran'][idx]

        return output_sequence

    def get_batch(self):
        if (self.current_position+self.total_length >= self.total()):
            return None
        data_dim = self.data['weather'].shape
        num_inputs = len(self.paths)
        batch= np.zeros((num_inputs, self.minibatch_size, self.total_length, data_dim[1])).astype(self.output_data_type)
        for i in range(self.minibatch_size):
            if (self.current_position + self.total_length >= self.total()):
                break
            self.current_batch = i
            input_seq = self.input_batch()
            output_seq = self.output_batch()
            #print(f'input_seq: {input_seq.shape}, output_seq: {output_seq.shape}')
            batch[:,i,:,:] = np.concatenate((input_seq, output_seq), axis=2).squeeze(axis=1)
            #print("Batch shape:", batch.shape)
            #self.update_batch_indices()
        batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2], 86, 86)
        batch = np.transpose(batch, (1, 2, 3, 4, 0))
        #print("Batch shape:", batch.shape)
        return batch


def main():
    weather_path = "../../data/datasets2/train/weather.csv"
    moran_path = "../../data/datasets2/train/moran.csv"
    input_param = {
        "paths": [weather_path, moran_path],
        "name": "test_input_handle",
        "minibatch_size": 6,
        "is_output_sequence": True,
        "input_length": 5,
        "total_length": 10,
        "overlap": 2,  # Define overlap for continuity
    }

    input_handle = InputHandle(input_param)
    input_handle.begin()

    while not input_handle.no_batch_left():
        batch = input_handle.get_batch()
        #print("Batch shape:", batch.shape)
        input_handle.next()


if __name__ == "__main__":
    main()
