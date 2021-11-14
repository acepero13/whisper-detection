from extensions.FileAudioDataGenerator import FileAudioDataGenerator


class Config:
    def __init__(self,
                 train_dir,
                 val_dir,
                 test_dir,
                 max_length_sec=2.4,
                 train_batch_size=32,
                 val_test_batch_size=32,
                 sample_rate=22050,
                 epochs=4,
                 model_path='/home/alvaro/Documents/ML/whispering/src/models/simple_new_model_more_data_v1_2.h5'

                 ):
        self.max_length = int(sample_rate * max_length_sec)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.input_shape = (self.max_length,)
        self.max_length_sec = max_length_sec
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size
        self.epochs = epochs
        self.sample_rate = sample_rate
        self.model_path = model_path

    def load_data(self):
        train_datagen = FileAudioDataGenerator(self.train_dir,
                                               batch_size=self.train_batch_size,
                                               max_length_secs=self.max_length_sec,
                                               class_mode='binary',
                                               shuffle=False)

        val_datagen = FileAudioDataGenerator(self.val_dir,
                                             batch_size=self.val_test_batch_size,
                                             max_length_secs=self.max_length_sec,
                                             class_mode='binary',
                                             shuffle=False)

        test_datagen = FileAudioDataGenerator(self.test_dir,
                                              batch_size=self.val_test_batch_size,
                                              max_length_secs=self.max_length_sec,
                                              class_mode='binary',
                                              shuffle=False)

        print(train_datagen.labels_dict, val_datagen.labels_dict, test_datagen.labels_dict)

        return train_datagen, val_datagen, test_datagen
