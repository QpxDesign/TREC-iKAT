def remove_extra_spaces(input_string):
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    cleaned_string = cleaned_string.replace("\n","")
    return cleaned_string

