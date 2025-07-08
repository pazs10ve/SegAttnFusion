import argparse
import torch
from experiments.train import train
from experiments.loaders import get_dataloaders
from src.model import get_model
from experiments.TrainingLogger import Logger
from torchvision.transforms import v2


def main():
    parser = argparse.ArgumentParser(description="Train/Test a model for classification.")
    parser.add_argument("--path", type=str, default="data", help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loaders.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test.")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment (optional).")
    parser.add_argument("--run_name", type=str, help="Name of the specific run (optional).")
    parser.add_argument("--log_dir", type=str, default='logs', help="Optional base directory for logs.")
    args = parser.parse_args()

    print("\n--- Script Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("--- End of Arguments ---\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger = Logger(
        experiment_name=args.experiment_name, 
        run_name=args.run_name,
        mode=args.mode, 
        base_dir=args.log_dir
    )
    logger.log_params(vars(args))

    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader, test_loader = get_dataloaders(data_dir=args.path, batch_size=args.batch_size, transform=transform)
    

    model = get_model(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "train":
        train_loader_list = list(train_loader)
        val_loader_list = list(val_loader)
        train_losses, val_losses = train(
                train_loader_list[:1400], val_loader_list[:400], model, criterion, optimizer, args.num_epochs, device
                )
        #train_losses, val_losses = train(
         #      train_loader, val_loader, model, criterion, optimizer, args.num_epochs, device
          #  )
   
        # Log metrics for each epoch
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            logger.log_metrics(epoch, train_loss=train_loss, val_loss=val_loss)
        
        logger.plot_losses(train_losses, val_losses)
        logger.save_model(model)
    elif args.mode == "test":
        test_loss = eval(test_loader, model, criterion, device)
        logger.log_metrics(epoch=0, test_loss=test_loss)

    logger.complete_run()


if __name__ == "__main__":
    main()




"""train_loader_list = list(train_loader)
        val_loader_list = list(val_loader)
        train_losses, val_losses = train(
                train_loader_list[:10], val_loader_list[:10], model, criterion, optimizer, args.num_epochs, device
                )"""