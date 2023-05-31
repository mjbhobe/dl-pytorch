"""
cmd_opts.py - parse command line options
"""
import argparse
import logging


def parse_command_line(help_banner=""):
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(help_banner)

    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="Flag that specifies that model should be trained (default=False)",
    )
    parser.set_defaults(train=False)

    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        help="Flag that specifies that model should validate performance against a dataset (default=False)",
    )
    parser.set_defaults(eval=False)

    parser.add_argument(
        "--pred",
        dest="pred",
        action="store_true",
        help="Flag that specifies that model should run predictions on test data (default=True)",
    )
    parser.set_defaults(pred=True)

    parser.add_argument(
        "--show_sample",
        dest="show_sample",
        action="store_true",
        help="Flag that specefies that a sample from dataset should be shown (default=False)",
    )
    parser.set_defaults(show_sample=False)

    parser.add_argument(
        "--use_cnn",
        dest="use_cnn",
        action="store_true",
        help="Flag to choose CNN model over ANN",
    )
    parser.add_argument(
        "--no-use_cnn",
        dest="use_cnn",
        action="store_false",
        help="Flag to choose ANN model over CNN",
    )
    parser.set_defaults(use_cnn=False)

    # epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="No of epochs to train model on (default 25)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training & cross-validation (default 64)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate to use for optimizer (default 0.001)",
    )

    parser.add_argument(
        "--l2_reg",
        type=float,
        default=0.0005,
        help="Amount of L2 regularization to apply to optimizer (default=0.0005)",
    )

    args = parser.parse_args()

    # display information
    logger.info("Parsed command line parameters:")
    logger.info(f"  - args.train {args.train}")
    logger.info(f"  - args.eval {args.eval}")
    logger.info(f"  - args.pred {args.pred}")
    logger.info(f"  - args.show_sample {args.show_sample}")
    logger.info(f"  - args.use_cnn {args.use_cnn}")
    logger.info(f"  - args.epochs {args.epochs}")
    logger.info(f"  - args.batch_size {args.batch_size}")
    logger.info(f"  - args.lr {args.lr:,.2f}")
    logger.info(f"  - args.l2_reg {args.l2_reg:,.5f}")

    return args
