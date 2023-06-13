import abc


class Regulator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate_control(self, *args):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            *args, **kwargs: Additional positional and key-word arguments.
        Returns:
            us: optimal control
        """
        pass