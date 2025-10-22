"""Pydantic model for default configuration and validation."""

import subprocess
from typing import Optional, Union
import os
from typing import Literal
from utils import BaseSettings
# from models.alignn import ALIGNNConfig
# from models.alignn_atomwise import ALIGNNAtomWiseConfig
# from models.ealignn_atomwise import eALIGNNAtomWiseConfig

# import torch

# try:
#     VERSION = (
#         subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
#     )
# except Exception:
#     VERSION = "NA"
#     pass


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


TARGET_ENUM = Literal[
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
    "gap pbe",
    "e_form",
    "e_hull",
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "e_above_hull",
    "mu_b",
    "bulk modulus",
    "shear modulus",
    "elastic anisotropy",
    "U0",
    "HOMO",
    "LUMO",
    "R2",
    "ZPVE",
    "omega1",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U",
    "H",
    "G",
    "Cv",
    "A",
    "B",
    "C",
    "all",
    "target",
    "max_efg",
    "avg_elec_mass",
    "avg_hole_mass",
    "_oqmd_band_gap",
    "_oqmd_delta_e",
    "_oqmd_stability",
    "edos_up",
    "pdos_elast",
    "bandgap",
    "energy_total",
    "net_magmom",
    "b3lyp_homo",
    "b3lyp_lumo",
    "b3lyp_gap",
    "b3lyp_scharber_pce",
    "b3lyp_scharber_voc",
    "b3lyp_scharber_jsc",
    "log_kd_ki",
    "max_co2_adsp",
    "min_co2_adsp",
    "lcd",
    "pld",
    "void_fraction",
    "surface_area_m2g",
    "surface_area_m2cm3",
    "indir_gap",
    "f_enp",
    "final_energy",
    "energy_per_atom",
    "ead",
]

class ALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_atomwise", "n_alignn_atomwise", "t_alignn_atomwise"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    heads: int = 1  # number of attention heads， 在n_alignn_atomwise中用到
    atom_input_features: int = 92
    # atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 64
    # hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1
    graphwise_weight: float = 1.0  # 误差放大因子
    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    force_mult_natoms: bool = False
    energy_mult_natoms: bool = False  # 使用energy_mult_natoms时对总能量修正，en_out = out * natoms
    # include_pos_deriv: bool = False  # 是否计算位置梯度
    use_cutoff_function: bool = False
    inner_cutoff: float = 3  # Ansgtrom
    stress_multiplier: float = 1
    # add_reverse_forces: bool = True  # will make True as default soon
    lg_on_fly: bool = True  # will make True as default soon
    batch_stress: bool = True
    multiply_cutoff: bool = False
    exponent: int = 5  # 5
    use_penalty: bool = True  # 使用energy_mult_natoms时对总能量修正
    penalty_factor: float = 0.1
    penalty_threshold: float = 1

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = "NA"

    # dataset configuration
    dataset: Literal[
        "dft_3d",
        "jdft_3d-8-18-2021",
        "dft_2d",
        "megnet",
        "megnet2",
        "mp_3d_2020",
        "qm9",
        "qm9_dgl",
        "qm9_std_jctc",
        "user_data",
        "oqmd_3d_no_cfid",
        "edos_up",
        "edos_pdos",
        "qmof",
        "qe_tb",
        "hmof",
        "hpov",
        "pdbbind",
        "pdbbind_core",
        "tinnet_OH",
        "tinnet_O",
        "tinnet_N",
    ] = "dft_3d"  # 在train_dgl里用到，在train_alignn中未指定数据集时才会启用
    target: TARGET_ENUM = "target" # 如果是dataset中的训练集，需要改成对应值, user_data时为target
    target_key: str = "formation_energy_peratom"  # 目标属性名，dataset为user_data时设置
    # atomic_number对应的atom_input_features为1， cgcnn--92
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "cgcnn"  # 原子特征，默认为cgcnn
    neighbor_strategy: Literal[
        "k-nearest", "voronoi", "radius_graph", "radius_graph_jarvis"
    ] = "k-nearest"  # 邻居策略，默认为k-nearest
    id_tag: Literal["jid", "id", "_oqmd_entry_id"] = "jid"  # 不需要更改，如果想要更改数据标签要改train_alignn中的id_key

    # training configuration
    dtype: str = "float32"
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    # target_range: Optional[List] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    target_multiplication_factor: Optional[float] = None  # 训练目标放大因子
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 1e-05  # 权重衰减
    learning_rate: float = 1e-2
    filename: str = "sample"  # 任务输出文件前缀
    warmup_steps: int = 2000  # 控制onecycle中的pct_start，需在train_dgl中设置使用，默认pct_start=0.3
    criterion: Literal["mse", "l1", "poisson", "zig"] = "mse"  # 损失函数，train中已定义
    optimizer: Literal["adamw", "sgd"] = "adamw"  # 优化器
    scheduler: Literal["onecycle", "none"] = "onecycle"  # 学习率调度器
    pin_memory: bool = False  # 是否使用pin_memory
    save_dataloader: bool = False  # 是否保存dataloader,dataloader中包含无法序列化的内容，无法保存
    # write_checkpoint: bool = True  # 是否保存checkpoint
    write_predictions: bool = True  # 是否保存预测结果
    # store_outputs: bool = True
    # progress: bool = True
    # log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False  # 是否使用标准化和PCA
    use_canonize: bool = True       # 是否使用canonize(规范边缘表示)
    compute_line_graph: bool = True
    num_workers: int = 4  # 数据加载器的线程数
    cutoff: float = 8.0
    cutoff_extra: float = 3.0
    max_neighbors: int = 12
    keep_data_order: bool = True
    normalize_graph_level_loss: bool = False
    distributed: bool = False
    data_parallel: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath(".")
    use_lmdb: bool = True
    read_existing:bool = True  # 是否读取已存在的lmdb数据集，一般config修改时要改为false
    # alignn_layers: int = 4
    # gcn_layers: int =4
    # edge_input_features: int= 80
    # hidden_features: int= 256
    # triplet_input_features: int=40
    # embedding_features: int=64

    # model configuration
    model: ALIGNNAtomWiseConfig = ALIGNNAtomWiseConfig(name="alignn_atomwise")
