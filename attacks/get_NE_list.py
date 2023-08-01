"""
Adapted from https://github.com/JHL-HUST/PWWS/blob/master/get_NE_list.py
"""
from collections import defaultdict

NE_type_dict = {
    "PERSON": defaultdict(int),  # People, including fictional.
    "NORP": defaultdict(int),  # Nationalities or religious or political groups.
    "FAC": defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    "ORG": defaultdict(int),  # Companies, agencies, institutions, etc.
    "GPE": defaultdict(int),  # Countries, cities, states.
    "LOC": defaultdict(int),  # Non-GPE locations, mountain ranges, bodies of water.
    "PRODUCT": defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    "EVENT": defaultdict(int),  # Named hurricanes, battles, wars, sports events, etc.
    "WORK_OF_ART": defaultdict(int),  # Titles of books, songs, etc.
    "LAW": defaultdict(int),  # Named documents made into laws.
    "LANGUAGE": defaultdict(int),  # Any named language.
    "DATE": defaultdict(int),  # Absolute or relative dates or periods.
    "TIME": defaultdict(int),  # Times smaller than a day.
    "PERCENT": defaultdict(int),  # Percentage, including "%".
    "MONEY": defaultdict(int),  # Monetary values, including unit.
    "QUANTITY": defaultdict(int),  # Measurements, as of weight or distance.
    "ORDINAL": defaultdict(int),  # "first", "second", etc.
    "CARDINAL": defaultdict(int),  # Numerals that do not fall under another type.
}


class NameEntityList:
    sst2_0 = {
        "PERSON": "spielberg",
        "NORP": "indian",
        "FAC": "broadway",
        "ORG": "parker",
        "GPE": "washington",
        "LOC": "marina",
        "PRODUCT": "discovery",
        "WORK_OF_ART": "stomp",
        "DATE": "daily",
        "TIME": "night",
        "ORDINAL": "20th",
        "CARDINAL": "2",
    }

    sst2_1 = {
        "PERSON": "lawrence",
        "NORP": "african-americans",
        "FAC": "jargon",
        "ORG": "showtime",
        "GPE": "brooklyn",
        "LOC": "valley",
        "PRODUCT": "akasha",
        "WORK_OF_ART": "horrible",
        "DATE": "1958",
        "TIME": "last-minute",
        "MONEY": "99",
        "ORDINAL": "7th",
        "CARDINAL": "zero",
    }

    imdb_0 = {
        "PERSON": "michael",
        "NORP": "australian",
        "FAC": "classic",
        "ORG": "columbo",
        "GPE": "india",
        "LOC": "atlantic",
        "PRODUCT": "ponyo",
        "EVENT": "war",
        "WORK_OF_ART": "batman",
        "LAW": "bible",
        "LANGUAGE": "japanese",
        "DATE": "2001",
        "TIME": "afternoon",
        "PERCENT": "98%",
        "MONEY": "1,000",
        "QUANTITY": "tons",
        "ORDINAL": "5th",
        "CARDINAL": "10/10",
    }

    imdb_1 = {
        "PERSON": "seagal",
        "NORP": "asian",
        "FAC": "fx",
        "ORG": "un",
        "GPE": "texas",
        "LOC": "dahmer",
        "PRODUCT": "concorde",
        "EVENT": "hugo",
        "WORK_OF_ART": "hamlet",
        "LAW": "thunderbirds",
        "LANGUAGE": "freakish",
        "DATE": "halloween",
        "TIME": "minutes",
        "PERCENT": "85%",
        "MONEY": "$",
        "QUANTITY": "two",
        "ORDINAL": "sixth",
        "CARDINAL": "4",
    }

    sst2 = [sst2_0, sst2_1]
    imdb = [imdb_0, imdb_1]

    L = {"imdb": imdb, "sst2": sst2}


NE_list = NameEntityList()
