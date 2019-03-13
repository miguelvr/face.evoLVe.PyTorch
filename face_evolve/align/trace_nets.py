import torch
from face_evolve.align.get_nets import PNet, RNet, ONet


def main():
    print("Tracing PNet")
    pnet_input = torch.rand(1, 3, 224, 224)
    pnet = PNet()
    pnet.eval()

    traced_pnet = torch.jit.trace(pnet, pnet_input)
    traced_pnet.save("pnet.pt")

    print("Tracing RNet")
    rnet_input = torch.rand(1, 3, 24, 24)
    rnet = RNet()
    rnet.eval()

    traced_pnet = torch.jit.trace(rnet, rnet_input)
    traced_pnet.save("rnet.pt")

    print("Tracing ONet")
    onet_input = torch.rand(1, 3, 48, 48)
    onet = ONet()
    onet.eval()

    traced_pnet = torch.jit.trace(onet, onet_input)
    traced_pnet.save("onet.pt")


if __name__ == '__main__':
    main()
