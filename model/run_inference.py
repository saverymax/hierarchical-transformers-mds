import torch

def main():
    with torch.no_grad():
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        total = 0
        for data in testloader:
            batch = iter(test_data)
            outputs = medel(input)

if __name__ == "__main__"
    main()
