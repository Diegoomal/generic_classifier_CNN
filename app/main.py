import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # needed for GPU access

from generic_classifier_CNN import GenericClassifierCNN


if __name__ == "__main__":

    gc_cnn = GenericClassifierCNN(path_dataset='../dataset')

    gc_cnn.verify_gpu()
    gc_cnn.test_dataset()
    
    hist, model = gc_cnn.run_train()

    gc_cnn.predict(model, '/tests/', filename='0.jpg')
    gc_cnn.predict(model, '/tests/', filename='1.png')
    gc_cnn.predict(model, '/tests/', filename='2.jpg')
