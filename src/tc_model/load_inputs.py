import h5py

def load_inputs(file_path):
    with h5py.File(file_path, 'r') as file:
        return file['train_genesis'][0]
