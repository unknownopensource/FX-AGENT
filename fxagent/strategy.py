def wants_long(completed_closes):
    """Demonstration rule, not a validated trading strategy.

    Receives only two already-completed closes.
    """
    return (
        len(completed_closes) == 2
        and completed_closes[-1] > completed_closes[-2]
    )
