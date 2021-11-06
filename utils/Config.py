class Config:
    def __init__(self,
                 train_dir,
                 val_dir,
                 test_dir,
                 normalize=True,
                 rms_level=0,
                 normalization_technique='min_max',
                 max_length_sec=4.2,
                 train_batch_size=32,
                 val_test_batch_size=32,
                 input_shape=(92610,),
                 epochs=4,
                 model_path='/home/alvaro/Documents/ML/whispering/src/models/simple_new_model_v1_1.h5'

                 ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.normalize = normalize
        self.input_shape = input_shape
        self.rms_level = rms_level
        self.normalization_technique = normalization_technique
        self.max_length_sec = max_length_sec
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size
        self.epochs = epochs
        self.model_path = model_path
