import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import BUSIDataset
from src.model import UNeXt
from src.losses import DiceBCELoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Train/validation split ---
    full_dataset = BUSIDataset('data/Dataset_BUSI_with_GT')
    val_size = int(0.2 * len(full_dataset))       # 20% held back
    train_size = len(full_dataset) - val_size      # 80% for training
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # --- Model, loss, optimizer ---
    model = UNeXt(num_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20  # keep small for a first real test; we'll increase later on Colab

    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()   # tell BatchNorm/Dropout layers "we're learning now"
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()          # clear old gradients
            preds = model(images)          # forward pass
            loss = criterion(preds, masks) # how wrong were we
            loss.backward()                # compute gradients (autograd)
            optimizer.step()               # apply the nudges

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation phase ----
        model.eval()   # tell BatchNorm/Dropout layers "we're just checking now, don't learn"
        val_loss = 0.0

        with torch.no_grad():   # don't waste time/memory computing gradients here
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), "results/unext_busi.pth")
    print("Model saved to results/unext_busi.pth")

if __name__ == "__main__":
    main()