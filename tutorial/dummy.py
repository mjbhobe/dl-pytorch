from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import time  # For simulation purposes


# Dummy training and validation loop to simulate the process
def training_and_validation_loop(total_epochs, training_batches, validation_batches):
    for epoch in range(1, total_epochs + 1):
        # Create a progress instance for each epoch
        total_batches = training_batches + validation_batches
        l_total_batches = len(str(total_batches))

        with Progress(
            TextColumn(f"Epoch ({epoch}/{total_epochs}): [bold green]"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(bar_width=30, complete_style="green"),
            # TimeRemainingColumn(),
            TextColumn(
                " ({task.fields[curr_batch]:5d}/{task.fields[total_batches]:5d})"
            ),
            TextColumn(" Loss: [bold red]{task.fields[loss]:.4f}"),
            TextColumn(" - Acc: [bold cyan]{task.fields[acc]:.2f}%"),
            TextColumn(" - Val_Loss: [bold yellow]{task.fields[val_loss]:.4f}"),
            TextColumn(" - Val_Acc: [bold magenta]{task.fields[val_acc]:.2f}%"),
        ) as progress:

            # Add a task for the entire epoch (training + validation)
            curr_batch = 0
            task = progress.add_task(
                "Training & Validation",
                total=total_batches,
                total_batches=total_batches,
                curr_batch=curr_batch + 1,
                loss=0.0,
                acc=0.0,
                val_loss=0.0,
                val_acc=0.0,
            )

            # ---- Training Phase ----
            for batch in range(1, training_batches + 1):
                # Simulate training metrics (replace with actual computations)
                current_loss = 0.1 * batch
                current_acc = 90.0 + (0.1 * batch)
                curr_batch += 1

                # Update the progress bar for training
                progress.update(
                    task,
                    advance=1,  # Increment the progress bar
                    total_batches=total_batches,
                    curr_batch=curr_batch,
                    loss=current_loss,
                    acc=current_acc,
                    val_loss=0.0,  # No validation metrics during training
                    val_acc=0.0,
                )

                # Simulate processing time for each training batch
                time.sleep(0.1)

            # ---- Validation Phase ----
            for batch in range(1, validation_batches + 1):
                # Simulate validation metrics (replace with actual computations)
                current_val_loss = 0.15 * batch
                current_val_acc = 85.0 + (0.05 * batch)
                curr_batch += 1

                # Update the progress bar for validation
                progress.update(
                    task,
                    advance=1,  # Increment the progress bar
                    curr_batch=curr_batch,
                    loss=0.0,  # No training metrics during validation
                    acc=0.0,
                    val_loss=current_val_loss,
                    val_acc=current_val_acc,
                )

                # Simulate processing time for each validation batch
                time.sleep(0.1)


# Simulate training with dummy values
training_and_validation_loop(total_epochs=3, training_batches=10, validation_batches=5)
