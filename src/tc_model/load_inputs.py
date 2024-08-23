import h5py

def load_inputs(file_path):
    with h5py.File(file_path, 'r') as file:
        return file['train_genesis'][0]

#def inputs_generator(file_path, dataset="train"):
#    with h5py.File(file_path, "r") as file:
#        geneses = file[dataset + "_genesis"]
#        for genesis in geneses:
#            yield genesis