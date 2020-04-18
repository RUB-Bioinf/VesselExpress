import csv


# save segment stats as csv file
def saveDictAsCSV(dictionary, path, header):
    """
    Save a dictionary as csv file

    Parameters
    ----------
    dictionary: dict
    path: string of path to save csv of dict
    header: list of csv header
    """
    list = [header]
    i = 0
    for item in dictionary:
        i = i + 1
        list_item = [i, item, dictionary[item]]
        list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list)
