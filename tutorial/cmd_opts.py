"""
cmd_opts.py - parse command line options
"""
import argparse
import logging

logger = logging.getLogger(__name__)


def parse_command_line():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser("Pytorch Fashion MNIST multi-class classifier")
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="Flag that specefied is model should be trained (true) or not",
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
    )
    parser.set_defaults(train=True)
    parser.add_argument(
        "--pred",
        dest="pred",
        action="store_true",
        help="Flag that specefied is model should be run predictions (true) or not",
    )
    parser.add_argument(
        "--no-pred",
        dest="pred",
        action="store_false",
    )
    parser.set_defaults(pred=False)
    parser.add_argument(
        "--show_sample",
        dest="show_sample",
        action="store_true",
        help="Flag that specefied if dataset sample should be shown (True) or not",
    )
    parser.add_argument(
        "--no-show_sample",
        dest="show_sample",
        action="store_false",
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
    )
    parser.set_defaults(use_cnn=False)

    # epochs
    parser.add_argument("--epochs", type=int, default=25, help="No of epochs to train model on")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training & cross-validation")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to use for optimizer")

    parser.add_argument(
        "--l2_reg", type=float, default=0.0005, help="Amount of L2 regularization to apply to optimizer"
    )

    args = parser.parse_args()

    # display information
    logger.info("Command line parameters:")
    logger.info(f"  - args.train {args.train}")
    logger.info(f"  - args.pred {args.train}")
    logger.info(f"  - args.show_sample {args.show_sample}")
    logger.info(f"  - args.use_cnn {args.use_cnn}")
    logger.info(f"  - args.epochs {args.epochs}")
    logger.info(f"  - args.batch_size {args.batch_size}")
    logger.info(f"  - args.lr {args.lr:,.2f}")
    logger.info(f"  - args.l2_reg {args.l2_reg:,.5f}")

    return args


class TrainingArgsParser(argparse.ArgumentParser):
    def __init__(
        self,
        prog=None,
        usage=None,
        description=None,
        epilog=None,
        parents=[],
        formatter_class=argparse.HelpFormatter,
        prefix_chars="-",
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler="error",
        add_help=True,
        allow_abbrev=True,
        exit_on_error=True,
    ):
        super(TrainingArgsParser, self).__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            parents=parents,
            formatter_class=formatter_class,
            prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler,
            add_help=add_help,
            allow_abbrev=allow_abbrev,
            exit_on_error=exit_on_error,
        )
        self.logger = logging.getLogger(__name__)
        self.args = None

        # add our command line arguments

        # --train flag to indicate model should be trained
        # default = False (do not train model!)
        self.add_argument(
            "--train",
            dest="train",
            action="store_true",
            help="Flag that specifies that model should be trained (default=False)",
        )
        self.set_defaults(train=False)

        # --eval flag to indicate model should evaluates for metrics post training
        # default = False (do not evaluate model!)
        self.add_argument(
            "--eval",
            dest="eval",
            action="store_true",
            help="Flag that specifies that model should validate performance against a dataset " "(default=False)",
        )
        self.set_defaults(eval=False)

        # --pred flag to indicate model should generate predictions
        # default = False (do not generate predictions)
        self.add_argument(
            "--pred",
            dest="pred",
            action="store_true",
            help="Flag that specifies that model should run predictions on test data (" "default=True)",
        )
        self.set_defaults(pred=False)

        # NOTE: by default, model will do nothing. Please pass either --train or --pred on
        # command line
        # if you pass --pred without first passing --train, model state will not be found!

        self.add_argument(
            "--show_sample",
            dest="show_sample",
            action="store_true",
            help="Flag that specefies that a sample from dataset should be shown (default=False)",
        )
        self.set_defaults(show_sample=False)

        # epochs
        self.add_argument(
            "--epochs",
            type=int,
            default=25,
            help="No of epochs to train model on (default 25)",
        )

        self.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="Batch size for training & cross-validation (default 64)",
        )

        self.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Learning rate to use for optimizer (default 0.001)",
        )

        self.add_argument(
            "--l2_reg",
            type=float,
            default=0.0005,
            help="Amount of L2 regularization to apply to optimizer (default=0.0005)",
        )

        self.add_argument(
            "--verbose",
            type=int,
            default=2,
            help="Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch (default 1).",
        )

        self.add_argument(
            "--val_split",
            type=float,
            default=0.20,
            help="Percentage of train set to use for cross-validation (default=0.2)",
        )

        self.add_argument(
            "--test_split",
            type=float,
            default=0.10,
            help="Percentage of train set to use for testing (default=0.1)",
        )

    def parse_args(self, args=None, namespace=None):
        self.args = super().parse_args(args, namespace)
        self.show_default_args()
        return self.args

    def show_default_args(self, print_args=False):
        # display information
        if print_args:
            print("Parsed command line parameters:")
            print(f"  - args.train {self.args.train}")
            print(f"  - args.eval {self.args.eval}")
            print(f"  - args.pred {self.args.pred}")
            print(f"  - args.show_sample {self.args.show_sample}")
            print(f"  - args.use_cnn {self.args.use_cnn}")
            print(f"  - args.epochs {self.args.epochs}")
            print(f"  - args.batch_size {self.args.batch_size}")
            print(f"  - args.lr {self.args.lr:,.6f}")
            print(f"  - args.l2_reg {self.args.l2_reg:,.6f}")
            print(f"  - args.val_split {self.args.val_split:,.6f}")
            print(f"  - args.test_split {self.args.test_split:,.6f}")
            print(f"  - args.verbose {self.args.verbose}")
        else:
            self.logger.info("Parsed command line parameters:")
            self.logger.info(f"  - args.train {self.args.train}")
            self.logger.info(f"  - args.eval {self.args.eval}")
            self.logger.info(f"  - args.pred {self.args.pred}")
            self.logger.info(f"  - args.show_sample {self.args.show_sample}")
            self.logger.info(f"  - args.use_cnn {self.args.use_cnn}")
            self.logger.info(f"  - args.epochs {self.args.epochs}")
            self.logger.info(f"  - args.batch_size {self.args.batch_size}")
            self.logger.info(f"  - args.lr {self.args.lr:,.6f}")
            self.logger.info(f"  - args.l2_reg {self.args.l2_reg:,.6f}")
            self.logger.info(f"  - args.val_split {self.args.val_split:,.6f}")
            self.logger.info(f"  - args.test_split {self.args.test_split:,.6f}")
            self.logger.info(f"  - args.verbose {self.args.verbose}")
