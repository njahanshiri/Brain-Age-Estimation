from  age_model_2D import age_model_2DCNN
from  age_model_3D import age_model_3DCNN
from  age_model_ECNN import age_model_ECNN




def age_model(model_type, level ="train", model_path = None, data_type=None, ax=None, log_pth=None):

    if model_type == "2DCNN":
        model = age_model_2DCNN(data_type, ax, log_pth)
        if level == "train":
            model.run()
        else:
            model.test(model_path)
    elif model_type == "3DCNN":
        model = age_model_3DCNN(data_type, log_pth)
        if level == "train":
            model.run()
        else:
            model.test(model_path)
    elif model_type == "ECNN":
        model = age_model_ECNN(log_pth)
        if level == "train":
            model.run()
        else:
            model.test(model_path)




