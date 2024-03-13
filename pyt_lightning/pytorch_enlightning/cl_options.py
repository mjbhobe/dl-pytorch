"""
cmd_opts.py - parse command line options
"""
import argparse
import logging


class TrainingArgsParser(argparse.ArgumentParser):
    """a custom command line args parser that setups some common
    command line options to train a Pytorch module.
    command line options available:
        --train (bool - default=False): specify to train the model
        --eval (bool - default=False): specify to evaluate model performance on a dataset post training
        --pred (bool - default=False): specify to run predictions
        --show_sample (bool - default=False): specify to show random sample from test dataset
        --epochs=N (int - default=25): specify the number of epochs for which module is trained
        --batch_size=N (int - default=64): specify the number batch size to use
        --lr=f (float - default=0.001): specify the optimizer's learning rate
        --l2_reg=f (float - default=0.0005): specify the L2 regularization to apply to optimizer
    """

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

        # self.add_argument(
        #     "--use_cnn",
        #     dest="use_cnn",
        #     action="store_true",
        #     help="Flag to choose CNN model over ANN",
        # )
        # self.add_argument(
        #     "--no-use_cnn",
        #     dest="use_cnn",
        #     action="store_false",
        #     help="Flag to choose ANN model over CNN",
        # )
        # self.set_defaults(use_cnn=False)

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

    def parse_args(self, args=None, namespace=None):
        """parse the command line arguments
        NOTE: between the constructor call & parse_args() call, you can add
        additional command line options as appropriate
        """
        self.args = super().parse_args(args, namespace)
        return self.args

    def show_parsed_args(self, print_args=False):
        """display or log parsed command line arguments & their values
        @params:
            - print_args=X (boolean, default=False) - if true prints parsed args to console,
                if false, logs the value as log.info(...) calls
        """
        # display information
        if print_args:
            print("Parsed command line parameters:")
            print(f"  - args.train {self.args.train}")
            print(f"  - args.eval {self.args.eval}")
            print(f"  - args.pred {self.args.pred}")
            print(f"  - args.show_sample {self.args.show_sample}")
            print(f"  - args.epochs {self.args.epochs}")
            print(f"  - args.batch_size {self.args.batch_size}")
            print(f"  - args.lr {self.args.lr:,.6f}")
            print(f"  - args.l2_reg {self.args.l2_reg:,.6f}")
        else:
            self.logger.info("Parsed command line parameters:")
            self.logger.info(f"  - args.train {self.args.train}")
            self.logger.info(f"  - args.eval {self.args.eval}")
            self.logger.info(f"  - args.pred {self.args.pred}")
            self.logger.info(f"  - args.show_sample {self.args.show_sample}")
            self.logger.info(f"  - args.epochs {self.args.epochs}")
            self.logger.info(f"  - args.batch_size {self.args.batch_size}")
            self.logger.info(f"  - args.lr {self.args.lr:,.6f}")
            self.logger.info(f"  - args.l2_reg {self.args.l2_reg:,.6f}")


def parse_command_line(help_banner=""):
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(help_banner)

    # --train flag to indicate model should be trained
    # default = False (do not train model!)
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="Flag that specifies that model should be trained (default=False)",
    )
    parser.set_defaults(train=False)

    # --eval flag to indicate model should evaluates for metrics post training
    # default = False (do not evaluate model!)
    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        help="Flag that specifies that model should validate performance against a dataset (" "default=False)",
    )
    parser.set_defaults(eval=False)

    # --pred flag to indicate model should generate predictions
    # default = False (do not generate predictions)
    parser.add_argument(
        "--pred",
        dest="pred",
        action="store_true",
        help="Flag that specifies that model should run predictions on test data (default=True)",
    )
    parser.set_defaults(pred=False)

    # NOTE: by default, model will do nothing. Please pass either --train or --pred on command
    # line
    # if you pass --pred without first passing --train, model state will not be found!

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
