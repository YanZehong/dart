def trip_advisor_label_map(rating):
    return int(rating)-1

def beer_advocate_label_map(rating):
    return int(rating)-1

def persent_label_map(label):
    return 1 if label == 2 else 0
        
def truncate(x, max_len):
    return x[:max_len]