from datetime import datetime, timedelta
import re
import base64

    
def get_current_datetime_for_filename():
    now = datetime.now()
    # Format the date and time as a string, using '-' and '_' to avoid invalid file name characters
    formatted_datetime = now.strftime("%Y-%m-%d_%I%M_%p")  # Use %I for 12-hour format and %p for AM/PM
    file_time = f"({formatted_datetime})"
    return file_time

def easy_time(seconds):
    seconds = int(seconds)
    hour, remainder = divmod(seconds, 3600)
    minute, seconds = divmod(remainder, 60)

    parts = []
    if hour:
        if hour==1:
            parts.append(f"{hour} HR")
        else:
            parts.append(f"{hour} HRS")
    if minute:
        parts.append(f"{minute} MIN")
    if seconds:
        parts.append(f"{seconds} SEC")

    return " ".join(parts)

def easy_eta(seconds):
    eta_time = timedelta(seconds=seconds)
    
    current_time = datetime.now()
    final_time = current_time + eta_time
    
    return final_time.strftime("%I:%M:%S %p")


def normalize_file_name(name):
    # Lowercase, remove underscores, hyphens, and spaces
    return re.sub(r'[_\-\s]', '', name.lower())

def join_with_and(items):
    # Return a string of items separated by commas, with 'and' before the last item.
    if not items:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')



