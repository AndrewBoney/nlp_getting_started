"""
preprocess_text
"""

def preprocess_text(text):
    # Make lower case
    text = text.str.lower()

    # Replace mentions with "@mention"
    text = text.str.replace(r"(@\S+)", " @mention", regex=True)

    # Replace link with "?link"
    text = text.str.replace(r"(http://\S+)|(https://\S+)", " ?link", regex=True)

    # Replace \n with space
    text = text.str.replace("\n", " ", regex=False)
    
    return text