import arrow

MAX_EPOCH_TIME = 2 ** 31 - 1


def iso_range_to_ts_range(start_date: str, end_date: str, tz: str):
    """
    Returns a tuple of (start_ts, end_ts) as epoch timestamps, given start_date and end_date in
    ISO format and the timezone mode in which the dates should be resolved to timestamps ('utc' or 'local')
    """
    start_ts, end_ts = None, MAX_EPOCH_TIME
    if start_date is not None:
        if tz == 'utc':
            start_ts = arrow.get(start_date).timestamp()
        elif tz == 'local':
            start_ts = arrow.get(start_date, tzinfo='local').timestamp()
    if end_date is not None:
        if tz == 'utc':
            end_ts = arrow.get(end_date).replace(
                hour=23, minute=59, second=59).timestamp()
        elif tz == 'local':
            end_ts = arrow.get(end_date, tzinfo='local').replace(
                hour=23, minute=59, second=59).timestamp()
    return (start_ts, end_ts)


def iso_to_date_only(*iso_strs: str):
    """
    For each ISO date string in the input, returns only the date part in the format 'YYYY-MM-DD'
    e.g. '2021-01-01T00:00:00.000Z' -> '2021-01-01'
    """
    return [iso_str[:10] if iso_str else None for iso_str in iso_strs]
