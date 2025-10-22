#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import os
import torch.distributed as dist
import csv
import sys
import json
import zipfile
from data import get_train_val_loaders
from train import train_dgl
from config import TrainingConfig, ALIGNNAtomWiseConfig
from jarvis.db.jsonutils import loadjson
import argparse
import torch
import time
from jarvis.core.atoms import Atoms
import random
# from ase.stress import voigt_6_to_full_3x3_stress


# 控制模型选择，需与config.model.name一致


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def setup(rank=0, world_size=0, port="12356"):
    """Set up multi GPU rank."""
    # "12356"
    if port == "":
        port = str(random.randint(10000, 99999))
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        # os.environ["MASTER_PORT"] = "12355"
        # Initialize the distributed environment.
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup(world_size):
    """Clean up distributed process."""
    if world_size > 1:
        dist.destroy_process_group()


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
)
parser.add_argument(
    "--root_dir",
    default="./data_qm9",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="./config/config.json",
    help="Name of the config file",
)
parser.add_argument(
    "--id_key",
    default="id",
    help="Name of the key for graph level id such as id",
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Folder to save outputs",
)
parser.add_argument(
    "--target_key",
    default=None,
    help="Name of the key for graph level data such as total_energy",
)
parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

# parser.add_argument(
#    "--keep_data_order",
#    default=True,
#    help="Whether to randomly shuffle samples",
# )

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)
parser.add_argument(
    "--restart_model_path",
    default=None,
    help="Checkpoint file path for model",
)
parser.add_argument(
    "--device",
    default=None,
    help="set device for training the model [e.g. cpu, cuda, cuda:2]",
)


def train_for_folder(
    rank=0,
    world_size=0,
    root_dir="examples/sample_data",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="total_energy",
    file_format="poscar",
    restart_model_path=None,
    output_dir=None,
):
    """Train for a folder."""
    setup(rank=rank, world_size=world_size)
    print("root_dir", root_dir)
    id_prop_json = os.path.join(root_dir, "id_prop.json")
    id_prop_json_zip = os.path.join(root_dir, "id_prop.json.zip")
    id_prop_csv = os.path.join(root_dir, "id_prop.csv")
    id_prop_csv_file = False
    multioutput = False
    # lists_length_equal = True
    if os.path.exists(id_prop_json_zip):
        dat = json.loads(
            zipfile.ZipFile(id_prop_json_zip).read("id_prop.json")
        )
    elif os.path.exists(id_prop_json):
        dat = loadjson(id_prop_json)
        print("----->read data from ", id_prop_json)
    elif os.path.exists(id_prop_csv):
        id_prop_csv_file = True
        with open(id_prop_csv, "r") as f:
            reader = csv.reader(f)
            dat = [row for row in reader]
        print("id_prop_csv_file exists", id_prop_csv_file)
    else:
        print("Check dataset file.")

    config_dict = loadjson(config_name)
    config = TrainingConfig(**config_dict)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
    # config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    if target_key is None:
        target_key = config.target_key

    n_outputs = []
    dataset = []
    # num = 0
    for i in dat:
        # num += 1
        # if num > 80000:
        #     break
        info = {}
        if id_prop_csv_file:
            file_name = i[0]
            tmp = [float(j) for j in i[1:]]  # float(i[1])
            info["jid"] = file_name

            if len(tmp) == 1:
                tmp = tmp[0]
            else:
                multioutput = True
                n_outputs.append(tmp)
            info["target"] = tmp
            file_path = os.path.join(root_dir, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                # Note using 500 angstrom as box size
                # Recommended install pytraj
                # conda install -c ambermd pytraj
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(
                    "File format not implemented", file_format
                )
            info["atoms"] = atoms.to_dict()
        else:
            info["target"] = i[target_key]
            info["atoms"] = i["atoms"]
            info["jid"] = i[id_key]
        dataset.append(info)
    print("----->len dataset: ", len(dataset))
    del dat

    line_graph = False
    if config.compute_line_graph:
        # if config.model.alignn_layers > 0:
        line_graph = True
    print("----->calc line graph: ", line_graph)

    if multioutput:
        print("----->multioutput", multioutput)
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]
        print("lists_length_equal", lists_length_equal, len(n_outputs[0]))
        if lists_length_equal:
            config.model.output_features = len(n_outputs[0])

        else:
            raise ValueError("Make sure the outputs are of same size.")

    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        line_graph=line_graph,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        batch_size=config.batch_size,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        root_dir=root_dir,
        use_lmdb=config.use_lmdb,
        read_existing=config.read_existing,
        dtype=config.dtype,
        split_seed=config.random_seed,
        # use_ddp: bool = False,
        # rank = rank,
        # world_size = world_size,
    )
    del dataset
    # print("dataset", dataset[0])
    t1 = time.time()
    # world_size = torch.cuda.device_count()
    print("rank", rank)
    print("world_size", world_size)
    train_dgl(
        config=config,
        restart_model_path=restart_model_path,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
        rank=rank,
        world_size=world_size,
    )
    t2 = time.time()
    t = t2 - t1
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    print("Time taken (s): ", f"{h: .0f}h:{m: .0f}m:{s: .2f}s")

    # train_data = get_torch_dataset(


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    world_size = int(torch.cuda.device_count())
    print("world_size", world_size)
    if world_size > 1:
        torch.multiprocessing.spawn(
            train_for_folder,
            args=(
                world_size,
                args.root_dir,
                args.config_name,
                args.classification_threshold,
                args.batch_size,
                args.epochs,
                args.id_key,
                args.target_key,
                args.file_format,
                args.restart_model_path,
                args.output_dir,
            ),
            nprocs=world_size,
        )
    else:
        train_for_folder(
            rank=0,
            world_size=world_size,
            root_dir=args.root_dir,
            config_name=args.config_name,
            classification_threshold=args.classification_threshold,
            batch_size=args.batch_size,
            epochs=args.epochs,
            id_key=args.id_key,
            target_key=args.target_key,
            file_format=args.file_format,
            restart_model_path=args.restart_model_path,
            output_dir=args.output_dir,
        )
    try:
        cleanup(world_size)
    except Exception:
        pass
