import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import BUSIDataset

ds = BUSIDataset('data/Dataset_BUSI_with_GT')

# DataLoader = the Keras "flow_from_directory" equivalent for batching
loader = DataLoader(
    ds,
    batch_size=8,      # like batch_size= in flow_from_directory
    shuffle=True,       # like shuffle=True in flow_from_directory
    num_workers=2        # parallel loading workers, similar concept to Keras' use_multiprocessing
)

images, masks = next(iter(loader))
print("Batch image shape:", images.shape)   # expect (8, 3, 256, 256)
print("Batch mask shape:", masks.shape)     # expect (8, 1, 256, 256)

# visualize first sample in the batch
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(images[0].permute(1, 2, 0))   # CHW -> HWC for plotting
ax[0].set_title("Image")
ax[1].imshow(masks[0][0], cmap="gray")     # drop channel dim for grayscale plot
ax[1].set_title("Mask")
plt.savefig("results/dataset_check.png")
print("Saved to results/dataset_check.png")