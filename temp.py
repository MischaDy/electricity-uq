import pickle


def get_model():
    filepath = r'comparison_storage/models/base_model_nn_n35136_it300_nh2_hs50.pth'
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    print('success!')
    return model


if __name__ == '__main__':
    get_model()
