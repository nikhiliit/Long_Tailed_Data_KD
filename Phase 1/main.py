from cifar_trainer import CIFAR100Trainer

def main():
    # Initialize the trainer with long-tailed data
    trainer = CIFAR100Trainer(batch_size=2048*2, learning_rate=0.001, num_epochs=50)

    # Plot the long-tailed label distribution
    trainer.plot_long_tailed_distribution()

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()