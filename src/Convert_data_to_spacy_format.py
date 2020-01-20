import json
from Convert_data_to_flair_format import convertDataToFlair


# Since spacy doesn't take overlapping entities as inputs, we are not using the overlapping entities
def readData(path):
    # data = json.load(path)
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    TRAIN_DATA = []
    SLOTS_INFO = []
    for i, dataItem in enumerate(data):
        entitiesDict = {}
        entitiesDict["entities"] = []
        SLOTS_INFO.append(dataItem["slots"])
        for slotItem in dataItem["slots"]:
            flag = False
            for tup in entitiesDict["entities"]:
                if (slotItem['startIndex'] <= tup[1] and tup[0] <= slotItem['endIndex']) or \
                        (slotItem['startIndex'] >= tup[0] and slotItem['endIndex'] <= tup[1]):
                    flag = True
                    break
            if flag == False:
                entitiesDict['entities'].append(
                    (slotItem['startIndex'], slotItem['endIndex'], slotItem['slotName']))

        TRAIN_DATA.append((dataItem['text'], entitiesDict))

    return TRAIN_DATA, SLOTS_INFO


if __name__ == '__main__':
    PATH = "data/dev-refactored.json"
    DATA, SLOTS_INFO = readData(PATH)
    convertDataToFlair(DATA, SLOTS_INFO, train=False)
