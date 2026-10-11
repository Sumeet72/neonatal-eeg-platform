import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        x = batch["spectrogram"].to(
            device,
            non_blocking=True
        )

        y = batch["label"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(x)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * y.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        total_correct += (
            predictions == y
        ).sum().item()

        total_samples += y.size(0)

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    accuracy = (
        total_correct
        / max(total_samples, 1)
    )

    return average_loss, accuracy


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        x = batch["spectrogram"].to(
            device,
            non_blocking=True
        )

        y = batch["label"].to(
            device,
            non_blocking=True
        )

        outputs = model(x)

        loss = criterion(
            outputs,
            y
        )

        total_loss += (
            loss.item()
            * y.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        total_correct += (
            predictions == y
        ).sum().item()

        total_samples += y.size(0)

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    accuracy = (
        total_correct
        / max(total_samples, 1)
    )

    return average_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    patience=3
):
    """
    Train model with validation and early stopping.
    """

    best_val_loss = float("inf")

    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "Early stopping."
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, history


if __name__ == "__main__":

    print(
        "Trainer module loaded successfully."
    )