from ignite.contrib.handlers.tensorboard_logger import (
    GradsHistHandler, GradsScalarHandler, TensorboardLogger,
    WeightsHistHandler, WeightsScalarHandler, global_step_from_engine)
from ignite.engine import Engine, Events


def init_tensorboard_logger(trainer, evaluator, optimizer, all_train_metrics, all_val_metrics, output_dir, debug=False):
    tb_logger = TensorboardLogger(log_dir=f"{output_dir}/tensorboard")

    print(list(evaluator.state.metrics.keys()))
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="1_validation",
        metric_names=list(all_val_metrics.keys()),
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="2_training",
        metric_names=list(all_train_metrics.keys()),
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="2_training",
        output_transform=lambda x: x[0]["loss"],
    )

    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        tag="3_params"
    )

    # for debugging 
    if debug:
        # Attach the logger to the trainer to log model's weights norm after each iteration
        tb_logger.attach(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            log_handler=WeightsScalarHandler(network)
        )

        # Attach the logger to the trainer to log model's weights as a histogram after each epoch
        tb_logger.attach(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            log_handler=WeightsHistHandler(network)
        )

        # Attach the logger to the trainer to log model's gradients norm after each iteration
        tb_logger.attach(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            log_handler=GradsScalarHandler(network)
        )

        # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
        tb_logger.attach(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            log_handler=GradsHistHandler(network)
        )
    return tb_logger
