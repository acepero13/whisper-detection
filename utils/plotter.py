import matplotlib.pyplot as plt


def plot_loss(history):
    history_dict = history.history
    loss_value = history_dict['loss']
    val_loss_value = history_dict['val_loss']
    acc = history_dict['accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss_value, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_value, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_accuracy(history):
    history_dict = history.history
    val_acc_values = history_dict['val_accuracy']

    acc = history_dict['accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
