import torch


def generate_3d_directions(num_thetas, device):

    if num_thetas % 3 != 0:
        raise ValueError(
            f"Number of theta's should be divisible by three, you provided {num_thetas}"
        )

    V1 = torch.vstack(
        [
            torch.sin(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
            torch.cos(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
            torch.zeros_like(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
        ]
    )

    V2 = torch.vstack(
        [
            torch.sin(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
            torch.zeros_like(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
            torch.cos(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
        ]
    )

    V3 = torch.vstack(
        [
            torch.zeros_like(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
            torch.sin(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
            torch.cos(
                torch.linspace(0, 2 * torch.pi, num_thetas // 3, device=device)
            ),
        ]
    )

    return torch.hstack([V1, V2, V3])


def generate_2d_directions(num_thetas, device):

    return torch.vstack(
        [
            torch.sin(
                torch.linspace(0, 2 * torch.pi, num_thetas, device=device)
            ),
            torch.cos(
                torch.linspace(0, 2 * torch.pi, num_thetas, device=device)
            ),
        ]
    )
