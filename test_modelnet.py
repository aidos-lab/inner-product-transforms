from datasets.modelnet import DataModule, DataModuleConfig

dm = DataModule(DataModuleConfig())


# import trimesh


# def read_ply(path):
#     mesh = trimesh.load_mesh(path)
#     pos = torch.from_numpy(mesh.vertices).to(torch.float)
#     return Data(pos=pos)


# data = ream
