from model import Transolver_Irregular_Mesh_3D, Transolver_Structured_Mesh_3D

def get_model(args):
    model_dict = {
        'Transolver_Irregular_Mesh_3D': Transolver_Irregular_Mesh_3D,
        'Transolver_Structured_Mesh_3D': Transolver_Structured_Mesh_3D,
    }
    return model_dict[args.model]
