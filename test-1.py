from keras.datasets import mnist


if __name__ == "__main__":
    print('Start test')
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))
    print(test_labels)
