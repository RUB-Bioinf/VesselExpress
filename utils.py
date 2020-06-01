import csv


def saveSegmentDictAsCSV(dictionary, path, measurement, unit=""):
    """
        Save a dictionary with measurements as csv file

        Parameters
        ----------
        dictionary: defaultdict(dict)
        path: string of path to save csv of dict
        measurement: string of measurement
        unit: string of unit
        """
    list = [[measurement, "Unit", "Category", "FilamentID", "SegmentID"]]
    for filament in dictionary.keys():
        for branch in dictionary[filament]:
            val = dictionary[filament][branch]
            list_item = [val, unit, "Segment", filament, branch]
            list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list)

def saveFilamentDictAsCSV(dictionary, path, measurement, unit=""):
    list = [[measurement, "Unit", "Category", "FilamentID"]]
    for filament in dictionary.keys():
        val = dictionary[filament]
        list_item = [val, unit, "Filament", filament]
        list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list)

