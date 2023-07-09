from enum import Enum


class Model(str, Enum):
    pcBOT = "pcBOT"
    PointCAM = "PointCAM"
    MLP = "MLP"


class Dataset(str, Enum):
    MODELNET40 = "ModelNet40"
    MODELNET40FPS = "ModelNet40FPS"
    MODELNET10 = "ModelNet10"
    SHAPENET = "ShapeNet"
    SHAPENETFPS = "ShapeNetFPS"
    SCANOBJECTNN = "ScanObjectNN"
    MODELNET40FEWSHOT = "ModelNet40FewShot"


class ScanObjectNNMode(str, Enum):
    OBJ_BG = "obj_bg"
    OBJ_ONLY = "obj_only"
    PB_T50_RS = "pb_t50_rs"


class Optimizer(str, Enum):
    ADAMW = "adamw"
    SGD = "sgd"


class Scheduler(str, Enum):
    CONSTANT = "constant"
    COSINE = "cosine"
    ONE_CYCLE = "one_cycle"


class FFNLayer(str, Enum):
    MLP = "mlp"
    SWIGLU = "swiglu"
