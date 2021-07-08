


def read_imagenet_classnames(path:str):
    with open(path, "r") as f:
        temp = f.readlines()

    temp = [i.strip(" ").strip("\n").strip(", ").split(":") \
                        for i in temp]
    temp = {int(k):v.strip(" ").split(",") for k,v in temp}
    classes = []
    for i in temp:
        classes.append([k.strip(" ") for k in temp[i]])
    return classes