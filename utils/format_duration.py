def format(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    duration_str = ""
    if hours > 0:
        duration_str += f"{hours}hrs "
    if minutes > 0:
        duration_str += f"{minutes}min "
    if remaining_seconds > 0 or (hours == 0 and minutes == 0):
        duration_str += f"{remaining_seconds}sec"

    return duration_str
